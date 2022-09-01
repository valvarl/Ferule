from __future__ import annotations

import json
import typing as tp
from enum import Enum
from ordered_set import OrderedSet

from tvm import relay, transform
from tvm import autotvm, auto_scheduler
from tvm.ir import IRModule
from tvm.target import Target

from ..model_importer import ModelImporter


class Tuner(Enum):
    UNKNOWN = "unknown",
    ATVM = "atvm",
    ANSOR = "ansor"


class Layer:
    def __init__(self, hf: HandleFile) -> None:
        self.configs = []
        self.hf = hf
        self.tuner = self.hf.tuner

    def add_config(self, config: dict) -> None:
        self.configs.append(config)

    def create_task(self) -> None:
        if self.tuner == Tuner.ATVM:
            input = self.configs[0]['input']
            target, task_name, args, _ = input
            
            data, kernel, strides, padding, dilation, data_layout, _, dtype = args
            data = relay.var("data", shape=data[1], dtype=dtype)
            kernel_size = (kernel[1][2], kernel[1][3]) if data_layout == "NCHW" else (kernel[1][1], kernel[1][2])
            kernel = relay.var("kernel", shape=kernel[1], dtype=dtype)

            self.out = relay.nn.conv2d(data, kernel, strides=strides, padding=padding, dilation=dilation, kernel_size=kernel_size, 
                data_layout=data_layout, out_dtype=dtype)
            self.mod = IRModule.from_expr(self.out)
        
        elif self.tuner == Tuner.ANSOR:
            workload_key, target, _, target_host, _, _ = self.configs[0]['i'][0]
            
            for task in self.hf.tasks:
                if workload_key == task.workload_key:
                    self.task: auto_scheduler.SearchTask = task
                    return
            
            raise RuntimeError("Workload key %s not found in model %s." % (str(self.workload_key), self.hf.model))

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
        def tupleit(t):
            return tuple(map(tupleit, t)) if isinstance(t, (list, tuple)) else t
        if self.tuner == Tuner.ATVM:
            tgt, task_name, task_args, task_kwargs = self.configs[0]["input"]
            return hash(tupleit((tgt, task_name, task_args)))
        elif self.tuner == Tuner.ANSOR:
            return hash(tupleit(self.configs[0]["i"][0]))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


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
        self.tasks, self.task_weights = auto_scheduler.extract_tasks(
                self.mod["main"], self.params, target=self.target)

        self.full_layer_list: tp.List[Layer] = []
        with open(self.file) as inf:
            if self.tuner == Tuner.ATVM:
                layer, config_index = 0, 1
                for line in inf:
                    layer_config = json.loads(line)
                    if config_index > layer_config['config']['index']:
                        if self.full_layer_list != []:
                            self.full_layer_list[-1].create_task()
                        self.full_layer_list.append(Layer(self))
                        config_index = 0
                        layer += 1
                    else:
                        config_index += 1
                    self.full_layer_list[-1].add_config(layer_config)
            else:
                layer = None
                layer_index = 0
                for line in inf:
                    layer_config = json.loads(line)
                    if layer != layer_config['i'][0][0]:
                        layer = layer_config['i'][0][0]
                        if self.full_layer_list != []:
                            self.full_layer_list[-1].create_task()
                        self.full_layer_list.append(Layer(self))
                        layer_index += 1
                    self.full_layer_list[-1].add_config(layer_config)
        
        self.layers = OrderedSet()
        for layer in self.full_layer_list:
            self.layers.add(layer)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, inp: Layer) -> Layer:
        if inp in self.layers:
            return self.layers[self.layers.index(inp)]
        raise IndexError("The first log file specified contains a unique layer %r that does not appear in the logs %s" % (inp, self.file))
