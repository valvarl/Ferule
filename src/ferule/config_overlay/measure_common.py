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

    configs = range(len(layer.configs))
    if best is not None:
        _a = [layer.get_config_time(c) for c in range(len(layer.configs))]
        configs = np.argsort(_a)[:best]
    
    tmp = utils.tempdir()
    data = np.zeros(len(layer.configs))
    for config_idx in tqdm(configs,  desc='Layer %d' % index):
        with open(tmp.relpath('config.json'), 'w') as conf:
            json.dump(layer.configs[config_idx].config, conf)

        for i in range(3):
            with silence():
                if layer.tuner == Tuner.ATVM:
                    config = task.space.ConfigEntity.from_json_dict(layer.configs[config_idx]['config'])
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
                prof_res = executor.xbenchmark(args, layer.hf.dtype, tmp.relpath('config.so'))
                data[config_idx] = np.mean(prof_res)
                break
            except tvm._ffi.base.TVMError:
                print(f'Connection failed, {2 - i} attempts left...')
                executor._disconnect_tracker() 
        
        with open(osp.join(config_dir, target_cfg_name), '+a') as conf:
            if layer.tuner == Tuner.ATVM:
                layer.configs[config_idx]['result'][0] = prof_res.tolist()
            elif layer.tuner == Tuner.ANSOR:
                layer.configs[config_idx]['r'][0] = prof_res.tolist()
            json.dump(layer.configs[config_idx].config, conf)
    
    tmp.remove()
    print(f'Layer {index} inference data saved at: {osp.join(config_dir, target_cfg_name)}')

    # if estimate is not None:
    #     st = pd.read_csv(estimate)
    #     common = calculate_metric(st[st.config == stat.loc[0, 'config']], '')
    #     t_min, t_max = min(common['times_min']), max(common['times_min'])
    #     t_opt = stat[devices].loc[0]
    #     est = 1 - (t_opt - t_min) / (t_max - t_min)
    #     metric = common['best_metric']
    #     print(f'\tLayer {index}: t_opt={[round(t, 6) for t in t_opt]}, t_min={t_min:.6f}, t_max={t_max:.6f}, ' \
    #         f'metric={metric:.6f}, est={[round(i, 6) for i in est]}, mean_est={est.mean():.6f}')


def analize(
        host_to_target: tp.Dict[str, tp.Dict[str, HandleFile]], 
        verbose: bool = False
        ) -> tp.Tuple(tp.Dict[Layer, tp.Dict[Config, float]], tp.Dict[str, tp.Dict[Layer, float]]):
    hosts = list(host_to_target.keys())
    targets = list(host_to_target[hosts[0]].keys())

    # Calculate metric

    layer_to_target_bt: tp.Dict[Layer, tp.Dict[str, tp.List[float]]] = dict()

    for comp in (lambda x, y: x != y, lambda x, y: x == y):
        for host in hosts:
            for target in targets:
                if comp(host, target):
                    for layer in host_to_target[host][target].layers:
                        if layer not in layer_to_target_bt:
                            layer_to_target_bt[layer] = {target: [layer.get_best_time()]}
                        else:
                            layer_to_target_bt[layer][target].append(layer.get_best_time())

    config_times: tp.Dict[Layer, tp.Dict[Config, tp.Dict[str, float]]] = dict()

    for sample_layer in layer_to_target_bt:
        config_times[sample_layer] = dict()
        for host in host:
            for target in targets:
                layer = host_to_target[host][target][layer]
                if layer != sample_layer:
                    continue
                for sample_config in sample_layer.configs:
                    if sample_config not in config_times[sample_layer]:
                        config_times[sample_layer][sample_config] = dict()
                    config_found = False
                    for config in layer:
                        if config != sample_config:
                            continue
                        config_found = True
                        config_times[sample_layer][sample_config][target] = config.get_time()
                        break
                    assert config_found, "config not found"

    metric: tp.Dict[Layer, tp.Dict[Config, float]] = dict()

    for layer in config_times:
        metric[layer] = dict()
        target_bt = min(layer_to_target_bt[layer])
        for config in config_times[layer]:
            m = 0
            for target, time in config_times[layer][config].items():
                m += (time - target_bt[target]) / target_bt[target]
            metric[layer][config] = m

    # Calculate estimation 

    estimate: tp.Dict[str, tp.Dict[Layer, float]] = dict()

    best_configs: tp.Dict[Layer, Config] = {layer: sorted(zip(metric[layer].values(), metric[layer]))[0][1] for layer in metric}

    for target in targets:
        estimate[target] = dict()
        if verbose:
            print(f'[INFO]: Target {target} estimation')
        for i, layer in enumerate(layer_to_target_bt):
            target_times = layer_to_target_bt[layer][target]
            t_min, t_max = min(target_times), max(target_times)
            t_opt = best_configs[layer].get_time()
            est = 1 - (t_opt - t_min) / (t_max - t_min)
            estimate[target][layer] = est
            if verbose:
                print(f'layer={i}, t_min={t_min:.6f}, t_max={t_max:.6f}, t_opt={t_opt:.6f}, est={est:.6f}')

    return metric, estimate


@contextlib.contextmanager
def silence():
    sys.stdout, old = io.StringIO(), sys.stdout
    try:
        yield
    finally:
        sys.stdout = old
