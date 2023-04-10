import io
import os
import os.path as osp
import sys
import json
import contextlib
import typing as tp
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tvm
from tvm.autotvm import task
from tvm.contrib import utils

from ..executor import Executor
from .dispatcher import Config, Layer, Tuner, HandleFile
from .. import view_folder, log_dir

output_dir = osp.join(view_folder, "plots")
config_dir = osp.join(log_dir, "common")


def get_common_statistic(
    layer: Layer, 
    executor: Executor, 
    index: int, 
    best: tp.Optional[int] = None,
    ) -> None:

    if not osp.exists(config_dir):
        os.makedirs(config_dir)

    host_cfg_name = osp.basename(layer.hf.file)
    target_cfg_name = executor.key + '_' + host_cfg_name

    configs = range(len(layer))
    if best is not None:
        _a = [layer.get_config_time(c) for c in range(len(layer))]
        configs = np.argsort(_a)[:best]
    
    tmp = utils.tempdir()
    for config_idx in tqdm(configs,  desc='Layer %d' % index):
        with open(tmp.relpath('config.json'), 'w') as conf:
            json.dump(layer[config_idx].config, conf)

        for i in range(3):
            with silence():
                if layer.tuner == Tuner.ATVM:
                    config = task.space.ConfigEntity.from_json_dict(layer[config_idx]['config'])
                    with executor.target:
                        schedule, args = layer.task.instantiate(config)
                    mod = tvm.lower(schedule, args)
                    executor.compile_autotvm(mod, None, tmp.relpath('config.json'), tmp.path)
                    
                elif layer.tuner == Tuner.ANSOR:
                    try:
                        schedule, args = layer.task.apply_best(tmp.relpath('config.json'))
                    except RuntimeError:
                        break
                    mod = tvm.lower(schedule, args)
                    executor.compile_ansor(mod, None, tmp.relpath('config.json'), tmp.path)

            try:
                prof_res = executor.xbenchmark(args, layer.hf.dtype, tmp.relpath('config.so')) / 1000
                break
            except tvm._ffi.base.TVMError:
                print(f'Connection failed, {2 - i} attempts left...')
                executor._disconnect_tracker() 
        
        with open(osp.join(config_dir, target_cfg_name), '+a') as conf:
            if layer.tuner == Tuner.ATVM:
                layer[config_idx]['result'][0] = prof_res.tolist()
            elif layer.tuner == Tuner.ANSOR:
                layer[config_idx]['r'][0] = prof_res.tolist()
            json.dump(layer[config_idx].config, conf)
            conf.write('\n')
    
    tmp.remove()
    print(f'Layer {index} inference data saved at: {osp.join(config_dir, target_cfg_name)}')


def analyze_configs(
        host_to_target: tp.Dict[str, tp.Dict[str, HandleFile]], 
        verbose: bool = False
        ) -> tp.Tuple[tp.Dict[Layer, tp.Dict[str, tp.Dict[Config, float]]], tp.Dict[str, tp.Dict[Layer, float]]]:
    hosts = list(host_to_target.keys())
    targets = list(host_to_target[hosts[0]].keys())

    # Calculate metric

    layer_to_target_bt: tp.Dict[Layer, tp.Dict[str, float]] = dict()

    layers_skiped = False
    for comp in (lambda x, y: x != y, lambda x, y: x == y):
        for host in hosts:
            for target in targets:
                if comp(host, target):
                    for layer in host_to_target[host][target].layers:
                        if layer not in layer_to_target_bt:
                            if host == target:
                                layers_skiped = True
                                continue
                            layer_to_target_bt[layer] = dict()
                        if host == target:
                            layer_to_target_bt[layer][target] = layer.get_best_time()
    if layers_skiped:
        print('[INFO]: Unique layers from host configs are skipped')

    config_times: tp.Dict[Layer, tp.Dict[str, tp.Dict[Config, tp.Dict[str, float]]]] = dict()

    for sample_layer in layer_to_target_bt:
        config_times[sample_layer] = dict()
        for host in hosts:
            config_times[sample_layer][host] = dict()
            for target in targets:

                if host != target:
                    layer = host_to_target[host][target][sample_layer]
                    for config in layer:
                        if config not in config_times[layer][host]:
                            config_times[layer][host][config] = dict()
                        config_times[layer][host][config][target] = config.get_time()
                else:
                    sl = host_to_target[host][targets[(targets.index(target) + 1) % len(targets)]][sample_layer]
                    layer = host_to_target[host][target][sl]
                    for c in sl:
                        config = layer[layer.configs.index(c)]
                        if config not in config_times[layer][host]:
                            config_times[layer][host][config] = dict()
                        config_times[layer][host][config][target] = config.get_time()

    metric: tp.Dict[Layer, tp.Dict[str, tp.Dict[Config, float]]] = dict()

    for layer in config_times:
        metric[layer] = dict()
        for host in config_times[layer]:
            metric[layer][host] = dict()
            for config in config_times[layer][host]:
                m = 0
                for target, time in config_times[layer][host][config].items():
                    target_bt = layer_to_target_bt[layer][target]
                    m += (time - target_bt) / target_bt
                metric[layer][host][config] = m

    # Calculate estimation 

    estimate: tp.Dict[str, tp.Dict[Layer, float]] = dict()

    best_configs_metric: tp.Dict[Layer, tp.Dict[str, Config]] = {layer: {host: sorted([(m, config) for config, m in metric[layer][host].items()], key=lambda x: x[0])[0][1] \
                                                                         for host in metric[layer]} for layer in metric}
    best_config_hosts: tp.Dict[Layer, tp.Dict[str, Config]] = {layer: {host: host_to_target[host][host][layer].get_best_config() for host in config_times[layer]} for layer in config_times}

    for target in targets:
        estimate[target] = dict()
        if verbose:
            print(f'[INFO]: Target {target} estimation')
        for i, layer in enumerate(layer_to_target_bt):
            target_times = []
            m = []
            t_opt = []
            for host in metric[layer]:
                best_config = best_configs_metric[layer][host]
                m.append(metric[layer][host][best_config])
                t_opt.append(config_times[layer][host][best_config][target])
                target_times.append(config_times[layer][host][best_config_hosts[layer][host]][target])

            m, host, t_opt = sorted(zip(m, hosts, t_opt))[0]
            t_min, t_max = min(target_times), max(target_times)
            t_min, t_max = min(t_min, t_opt), max(t_max, t_opt)
            est = 1 - (t_opt - t_min) / (t_max - t_min)
            estimate[target][layer] = est
            if verbose:
                print(f'layer={i} host={host:<8} metric={m:.6f} t_min={t_min:.8f} t_opt={t_opt:.8f} t_max={t_max:.8f} est={est:.6f}')
    
    return metric, estimate


@contextlib.contextmanager
def silence():
    sys.stdout, old = io.StringIO(), sys.stdout
    try:
        yield
    finally:
        sys.stdout = old
