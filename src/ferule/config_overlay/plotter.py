import io
import os
import sys
import json
import contextlib
import typing as tp
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tvm
from tvm.autotvm import task
from tvm.contrib import utils

from ..executor import Executor
from .dispatcher import Layer, Tuner

output_dir = "output"


def draw_config_overlay_graph(layers: tp.List[Layer], labels: tp.List[str], index: int) -> None:
    data = [[None for j in range(len(layers))] for i in range(len(layers[0].configs))]
    if layers[0].tuner == Tuner.ATVM:
        tuner = "atvm"
        name = "%d.%s.%s" % (index, layers[0].configs[0]['input'][1], layers[0].configs[0]['input'][2][0][1])
        name = name.replace(" ", "")
        for layer_idx, layer in enumerate(layers):
            for config_idx, config in enumerate(layer.configs):
                if config['result'][1] == 0:  # if no errors occur
                    data[config_idx][layer_idx] = np.mean(config['result'][0])
    elif layers[0].tuner == Tuner.ANSOR:
        tuner = "ansor"
        name = "%d.input.%s" % (index, json.loads(layers[0].configs[0]['i'][0][0])[1][1:])
        name = name.replace(" ", "")
        for layer_idx, layer in enumerate(layers):
            for config_idx, config in enumerate(layer.configs):
                if config['r'][1] == 0 and config_idx < len(data):  # if no errors occur
                    data[config_idx][layer_idx] = np.mean(config['r'][0])
    
    df = pd.DataFrame(data, columns=range(len(layers))).sort_values(0) * 1000
    df.index = range(len(layers[0].configs))
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(name, fontsize=16)
    plt.xlabel("index", fontsize=14)
    plt.ylabel("time, ms", fontsize=14)
    plt.plot(df)
    plt.legend(labels, prop={'size': 12})
    if not os.path.exists(os.path.join(output_dir, tuner)):
        os.makedirs(os.path.join(output_dir, tuner))
    plt.savefig(os.path.join(output_dir, tuner, name + ".png"))
    plt.close()

def mesure_and_draw_co_graph(layer: Layer, executors: tp.Sequence[Executor], index: int) -> None:
    tmp = utils.tempdir()
    
    data = [[None for j in range(len(executors))] for i in range(len(layer.configs))]
    if layer.tuner == Tuner.ATVM:
        for config_idx in tqdm(range(len(layer.configs)),  desc='Layer %d' % index):
            with silence():
                with open(tmp.relpath('config.json'), 'w') as conf:
                    json.dump(layer.configs[config_idx], conf)
                config = task.space.ConfigEntity.from_json_dict(layer.configs[config_idx]['config'])
                with executors[0].target:
                    schedule, args = layer.task.instantiate(config)
                mod = tvm.lower(schedule, args)
                executors[0].compile_autotvm(mod, None, tmp.relpath('config.json'), tmp.path)
                for layer_idx, executor in enumerate(executors):
                    data[config_idx][layer_idx]  = executor.xbenchmark(args, layer.hf.dtype, tmp.relpath('config.so')) 
    
    if layer.tuner == Tuner.ANSOR:
        for config_idx in tqdm(range(len(layer.configs)),  desc='Layer %d' % index):
            with silence():
                with open(tmp.relpath('config.json'), 'w') as conf:
                    json.dump(layer.configs[config_idx], conf)
                try:
                    schedule, args = layer.task.apply_best(tmp.relpath('config.json'))
                except RuntimeError:
                    continue
                mod = tvm.lower(schedule, args)
                executors[0].compile_ansor(mod, None, tmp.relpath('config.json'), tmp.path)
                for layer_idx, executor in enumerate(executors):
                    data[config_idx][layer_idx]  = executor.xbenchmark(args, layer.hf.dtype, tmp.relpath('config.so'))           
    
    tmp.remove()
    df = pd.DataFrame(data, columns=[executor.key for executor in executors]).sort_values(executors[0].key)
    df.index = range(len(layer.configs))
    fig = plt.figure(figsize=(8, 6))
    name = "%d.%s" % (index, layer.name)
    fig.suptitle(name, fontsize=16)
    plt.xlabel("index", fontsize=14)
    plt.ylabel("time, ms", fontsize=14)
    plt.plot(df)
    plt.legend(df.columns, prop={'size': 12})
    if not os.path.exists(os.path.join(output_dir, *layer.tuner.value)):
        os.makedirs(os.path.join(output_dir, *layer.tuner.value))
    plt.savefig(os.path.join(output_dir, *layer.tuner.value, name + ".png"))
    plt.close()

@contextlib.contextmanager
def silence():
    sys.stdout, old = io.StringIO(), sys.stdout
    try:
        yield
    finally:
        sys.stdout = old
    