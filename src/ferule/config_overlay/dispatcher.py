from __future__ import annotations

import json
import typing as tp
from enum import Enum
from ordered_set import OrderedSet

import numpy as np
from tvm import autotvm, auto_scheduler
from tvm.target import Target

from ..model_importer import ModelImporter


def tupleit(t):
    return tuple(map(tupleit, t)) if isinstance(t, (list, tuple)) else t


class Tuner(Enum):
    UNKNOWN = "unknown",
    ATVM = "atvm",
    ANSOR = "ansor",


class Config:
    def __init__(self, config: tp.Dict, tuner: Tuner) -> None:
        self.config = config    
        self.tuner = tuner
    
    def __hash__(self) -> int:
        if self.tuner == Tuner.ATVM:
            tgt, task_name, task_args, task_kwargs = self.config["input"]
            return hash(tupleit((tgt, task_name, task_args)))
        elif self.tuner == Tuner.ANSOR:
            return hash(tupleit(self.config["i"][0]))
        
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
    
    def __len__(self) -> int:
        return len(self.config)

    def __getitem__(self, inp: tp.Any) -> tp.Any:
        return self.config[inp]
    
    def get_time(self):
        if self.tuner == Tuner.ATVM: 
            return np.mean(self.config['result'][0]) if self.config['result'][1] == 0 else 1e9
        elif self.tuner == Tuner.ANSOR:
            return np.mean(self.config['r'][0]) if self.config['r'][1] == 0 else 1e9


class Layer:
    def __init__(self, hf: HandleFile) -> None:
        self.configs: tp.List[Config] = []
        self.hf = hf
        self.tuner = self.hf.tuner

    def add_config(self, config: dict) -> None:
        self.configs.append(Config(config, self.tuner))

    def get_config_time(self, idx):
        return self.configs[idx].get_time()

    def get_best_time(self) -> float:
        return np.min([config.get_time() for config in self.configs])

    def create_task(self) -> None:
        if self.tuner == Tuner.ATVM:
            input = self.configs[0]['input']
            target, task_name, args, _ = input
            
            args = tupleit(args)
            for task in self.hf.tasks:
                if args == task.__dict__['args']:
                    self.task: autotvm.task.task.Task = task
                    return
        
        elif self.tuner == Tuner.ANSOR:
            workload_key, target, _, target_host, _, _ = self.configs[0]['i'][0]
            
            for task in self.hf.tasks:
                if workload_key == task.workload_key:
                    self.task: auto_scheduler.SearchTask = task
                    return
            
            raise RuntimeError("Workload key %s not found in model %s." % (str(workload_key), self.hf.model))

    @property
    def name(self) -> str:
        if self.tuner == Tuner.ATVM:
            name = "%s.%s" % (self.configs[0]['input'][1], self.configs[0]['input'][2][0][1])
            return name.replace(" ", "")
        elif self.tuner == Tuner.ANSOR:
            name = ".".join([str(j) for j in json.loads(self.configs[0]['i'][0][0])[1:]])
            return name.replace(" ", "")
        return self.tuner.name        

    def __hash__(self) -> int:
        return hash(self.configs[0])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
    
    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, idx) -> Config:
        return self.configs[idx]


class HandleFile:
    def __init__(self, file: str, model: tp.Optional[str] = None, dtype: str = "float32") -> None:
        self.file = file
        self.model = None
        for m in ModelImporter.available_models():
            if m in self.file:
                self.model = m
                break
        
        if self.model is None:
            if model is not None:
                self.model = model
            else:
                raise RuntimeError("Model must be specified.")

        self.dtype = None
        if "float32" in self.file:
            self.dtype = "float32"
        elif "float16" in self.file:
            self.dtype = "float16"
        else:
            self.dtype = dtype

        with open(self.file) as inf:
            self.tuner = Tuner.UNKNOWN
            line = json.loads(inf.readline())
            if 'i' in line and 'r' in line:
                self.tuner = Tuner.ANSOR
                self.workload_key, target, _, target_host, _, _ = line['i'][0]
                self.target = Target(target, host=target_host)
            elif 'input' in line and 'result' in line:
                self.tuner = Tuner.ATVM
                self.target = Target(line['input'][0])
            else:
                class FileFormatError(Exception):
                    def __init__(self, message) -> None:
                        super().__init__(message)

                raise FileFormatError("It is not possible to determine which tuner logs are being used.")
        self.__post_init__()

    def __post_init__(self):
        importer = ModelImporter()
        self.mod, self.params = importer(self.model, tuner="atvm" if self.tuner == Tuner.ATVM else "ansor", dtype=self.dtype)
        if self.tuner == Tuner.ATVM:
            self.tasks = autotvm.task.extract_from_program(
                self.mod["main"], target=self.target, params=self.params)
        else:
            self.tasks, self.task_weights = auto_scheduler.extract_tasks(
                    self.mod["main"], self.params, target=self.target)

        self.full_layer_list: tp.List[Layer] = []
        with open(self.file) as inf:

            for i, line in enumerate(inf):
                layer_config = json.loads(line)
                if self.tuner == Tuner.ATVM:
                    if i == 0:
                        self.full_layer_list.append(Layer(self))
                    elif layer_config['config']['index'] < last:
                        self.full_layer_list[-1].create_task()
                        self.full_layer_list.append(Layer(self))
                    self.full_layer_list[-1].add_config(layer_config)
                    last = layer_config['config']['index']

                elif self.tuner == Tuner.ANSOR:
                    if i == 0:
                        self.full_layer_list.append(Layer(self))
                    elif layer_config['i'][0][0] != last:
                        self.full_layer_list[-1].create_task()
                        self.full_layer_list.append(Layer(self))
                    self.full_layer_list[-1].add_config(layer_config)
                    last = layer_config['i'][0][0]
            
            if self.full_layer_list != []:
                self.full_layer_list[-1].create_task()
        
        self.layers: tp.Set[Layer] = OrderedSet()
        for layer in self.full_layer_list:
            self.layers.add(layer)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, inp: Layer) -> Layer:
        if inp in self.layers:
            return self.layers[self.layers.index(inp)]
        raise IndexError("The first log file specified contains a unique layer %r that does not appear in the logs %s" % (inp, self.file))
