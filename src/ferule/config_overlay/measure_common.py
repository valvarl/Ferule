import io
import os
import sys
import json
import contextlib
import typing as tp
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tvm
from tvm.autotvm import task
from tvm.contrib import utils

from ..executor import Executor
from .dispatcher import Layer, Tuner
from .. import view_folder, log_dir

output_dir = os.path.join(view_folder, "plots")
config_dir = os.path.join(log_dir, "common")


def get_fast_common_statistic(layers: tp.List[Layer], labels: tp.List[str], index: int) -> None:
    '''
    Useful only for AutoTVM gridsearch algorithm with same number of trials.
    '''
    data = [[None for j in range(len(layers))] for i in range(len(layers[0].configs))]
    if layers[0].tuner == Tuner.ATVM:
        for layer_idx, layer in enumerate(layers):
            for config_idx, config in enumerate(layer.configs):
                if config['result'][1] == 0:  # if no errors occur
                    data[config_idx][layer_idx] = np.mean(config['result'][0])
    
    elif layers[0].tuner == Tuner.ANSOR:
        for layer_idx, layer in enumerate(layers):
            for config_idx, config in enumerate(layer.configs):
                if config['r'][1] == 0 and config_idx < len(data):  # if no errors occur
                    data[config_idx][layer_idx] = np.mean(config['r'][0])
    
    df = pd.DataFrame(data, columns=labels).sort_values(0) * 1000
    visualize_common(layer, index, df)


def get_common_statistic(layers: Layer | list[Layer], executors: tp.Sequence[Executor], index: int) -> None:
    tmp = utils.tempdir()
    
    if isinstance(layers, tp.Iterable):
        layer = layers[0]
    else:
        layer = layers 

    data = [[None for j in range(len(executors))] for i in range(len(layer.configs))]
    for layer_idx, executor in enumerate(executors):
        print(f"Check the availability of {executor.key} device. ", end='')
        # input("Press Enter to continue...")
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
                    data[config_idx][layer_idx]  = executor.xbenchmark(args, layer.hf.dtype, tmp.relpath('config.so'))           
    
    tmp.remove()
    df = pd.DataFrame(data, columns=[executor.key for executor in executors])

    # calculate metric
    best_config = None
    if isinstance(layers, tp.Iterable):
        device_best_time = pd.DataFrame([[l.get_best_time() for l in layers]], columns=['best_' + executor.key for executor in executors]) * 1000
        logs = pd.concat([pd.Series([index] * len(df), name='layer'), pd.concat([device_best_time] * len(df), ignore_index=True), df], axis=1)
        device_best_time.columns = [i[5:] for i in device_best_time.columns]
        metric = df.sub(device_best_time.squeeze()).div(device_best_time.squeeze()).sum(axis=1)
        logs['metric'] = metric
        logs['config'] = [json.dumps(i) for i in layer.configs]
        
        name = f'{"_".join(device_best_time.columns)}.{layer.hf.model}.{layer.hf.dtype}.{layer.tuner.value[0]}'
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        log_file = os.path.join(config_dir, name + ".csv")
        logs.to_csv(log_file, mode='a', index=False, header=not os.path.isfile(log_file))
        print(f'Layer {index} inference data saved at: {log_file}')
    
        metric_sorted = metric[metric != 0].sort_values()
        df_sorted = df.iloc[metric_sorted.index]
        best_config = df_sorted.index[0]
        with open(os.path.join(config_dir, name + '.json'), 'a') as ouf:
            json.dump(layer.configs[best_config], ouf)
            ouf.write('\n')

        visualize_metric(layer, index, df_sorted, metric_sorted)

    df_sorted = df.sort_values(executors[0].key)
    visualize_common(layer, index, df_sorted, best_config)


def visualize_metric(layer: Layer, index: int, df: pd.DataFrame, metric: pd.Series) -> None:
    fig = plt.figure(figsize=(8, 6))
    name = "%d.%s" % (index, layer.name)
    fig.suptitle(name, fontsize=16)
    plt.xlabel("metric", fontsize=14)
    plt.ylabel("time, ms", fontsize=14)
    plt.plot(df.reset_index(drop=True), label=df.columns)
    plt.plot(metric.reset_index(drop=True), '--', color='black', label='metric')
    plt.legend(prop={'size': 12})
    plt.xticks(range(len(metric)), [f"{m:.2f}" for m in metric])
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(6))
    if not os.path.exists(os.path.join(output_dir, layer.tuner.value[0])):
        os.makedirs(os.path.join(output_dir, layer.tuner.value[0]))
    plt.savefig(os.path.join(output_dir, layer.tuner.value[0], name + "_metric.png"))
    plt.close()


def visualize_common(layer: Layer, index: int, df: pd.DataFrame, best_config: tp.Optional[int] = None) -> None:
    fig = plt.figure(figsize=(8, 6))
    name = "%d.%s" % (index, layer.name)
    fig.suptitle(name, fontsize=16)
    plt.xlabel("index", fontsize=14)
    plt.ylabel("time, ms", fontsize=14)
    plt.plot(df.reset_index(drop=True), label=df.columns)
    if best_config is not None:
        plt.axvline(x=np.arange(len(df))[df.index == best_config][0], color='black', ls='--', label='best config')
    plt.legend(prop={'size': 12})
    if not os.path.exists(os.path.join(output_dir, layer.tuner.value[0])):
        os.makedirs(os.path.join(output_dir, layer.tuner.value[0]))
    plt.savefig(os.path.join(output_dir, layer.tuner.value[0], name + ".png"))
    plt.close()


@contextlib.contextmanager
def silence():
    sys.stdout, old = io.StringIO(), sys.stdout
    try:
        yield
    finally:
        sys.stdout = old
