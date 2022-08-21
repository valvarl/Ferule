import json
import typing as tp
from enum import Enum
from ordered_set import OrderedSet



class Tuner(Enum):
    UNKNOWN = 0,
    ATVM = 1,
    ANSOR = 2


class Layer:
    def __init__(self, tuner=Tuner.UNKNOWN) -> None:
        self.tuner = tuner
        self.configs: dict = []

    def add_config(self, config: dict) -> None:
        self.configs.append(config)

    @property
    def name(self) -> str:
        return '%d.%s.%s' % (self.index, self.inp[1], self.inp[2][0][1])

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
    def __init__(self, file) -> None:
        self.file = file
        with open(self.file) as inf:
            self.tuner = Tuner.UNKNOWN
            line = json.loads(inf.readline())
            if 'i' in line and 'r' in line:
                self.tuner = Tuner.ANSOR
            elif 'input' in line and 'result' in line:
                self.tuner = Tuner.ATVM
            else:
                class FileFormatError(Exception):
                    def __init__(self, message) -> None:
                        super().__init__(message)

                raise FileFormatError("It is not possible to determine which tuner logs are being used.")
        self.__post_init__()

    def __post_init__(self):
        self.full_layer_list: tp.List[Layer] = []
        with open(self.file) as inf:
            if self.tuner == Tuner.ATVM:
                layer, config_index = 0, 1
                for line in inf:
                    layer_config = json.loads(line)
                    if config_index > layer_config['config']['index']:
                        self.full_layer_list.append(Layer(Tuner.ATVM))
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
                        self.full_layer_list.append(Layer(Tuner.ANSOR))
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
