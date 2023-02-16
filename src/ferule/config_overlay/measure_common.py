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
from .dispatcher import Layer, Tuner
from .. import view_folder, log_dir

output_dir = osp.join(view_folder, "plots")
config_dir = osp.join(log_dir, "common")


def get_common_statistic(
    layer: Layer, 
    executors: tp.Sequence[Executor], 
    index: int, 
    best: tp.Optional[int] = None,
    estimate: tp.Optional[Path] = None
    ) -> None:
    
    tmp = utils.tempdir()
    data = np.zeros((len(layer.configs), len(executors)))
    for exec_idx, executor in enumerate(executors):
        print(f"Check the availability of {executor.key} device. ", end='')
        # input("Press Enter to continue...")
        configs = range(len(layer.configs))
        
        if estimate is not None:
            best = 1     
        if best is not None:
            _a = [layer.get_config_time(c) for c in range(len(layer.configs))]
            configs = np.argsort(_a)[:best]

        for config_idx in tqdm(configs,  desc='Layer %d' % index):
            with open(tmp.relpath('config.json'), 'w') as conf:
                json.dump(layer.configs[config_idx], conf)

            for i in range(3):
                with silence():
                    if layer.tuner == Tuner.ATVM:
                        config = task.space.ConfigEntity.from_json_dict(layer.configs[config_idx]['config'])
                        with executors[0].target:
                            schedule, args = layer.task.instantiate(config)
                        mod = tvm.lower(schedule, args)
                        executors[0].compile_autotvm(mod, None, tmp.relpath('config.json'), tmp.path)
                        
                    if layer.tuner == Tuner.ANSOR:
                        try:
                            schedule, args = layer.task.apply_best(tmp.relpath('config.json'))
                        except RuntimeError:
                            break
                        mod = tvm.lower(schedule, args)
                        executors[0].compile_ansor(mod, None, tmp.relpath('config.json'), tmp.path)

                try:
                    data[config_idx][exec_idx]  = executor.xbenchmark(args, layer.hf.dtype, tmp.relpath('config.so'))
                    break
                except tvm._ffi.base.TVMError:
                    print(f'Connection failed, {2 - i} attempts left...')
                    executor._disconnect_tracker() 
    
    tmp.remove()
    devices = [executor.key for executor in executors]
    stat = pd.DataFrame(data, columns=devices)
    stat['config'] = [json.dumps(i) for i in layer.configs]
    stat['layer'] = index
    stat = stat[[stat.columns[-1]] + stat.columns[:-1].to_list()]

    name = f'{"_".join(devices)}.{layer.hf.model}.{layer.hf.dtype}.{layer.tuner.value[0]}'
    if not osp.exists(config_dir):
        os.makedirs(config_dir)
    log_file = osp.join(config_dir, name + (".csv" if estimate is None else f"_best.csv"))
    stat.to_csv(log_file, mode='a', index=False, header=not osp.isfile(log_file))
    print(f'Layer {index} inference data saved at: {log_file}')

    if estimate is not None:
        st = pd.read_csv(estimate)
        common = calculate_metric(st[st.config == stat.loc[0, 'config']], '')
        t_min, t_max = min(common['times_min']), max(common['times_min'])
        t_opt = stat[devices].loc[0]
        est = 1 - (t_opt - t_min) / (t_max - t_min)
        metric = common['best_metric']
        print(f'\tLayer {index}: t_opt={[round(t, 6) for t in t_opt]}, t_min={t_min:.6f}, t_max={t_max:.6f}, ' \
            f'metric={metric:.6f}, est={[round(i, 6) for i in est]}, mean_est={est.mean():.6f}')


def calculate_metric(
    stat: pd.DataFrame,
    name: str,
    vis: bool = False, 
    verbose: bool = False
    ) -> dict:
    
    devices = stat.columns[1: -1]
    stat = stat.loc[(stat[devices] > 0).all(axis=1)]
    device_best_time = stat[devices].min()

    pd.set_option('mode.chained_assignment', None)
    stat['metric']  = stat[devices].sub(device_best_time.squeeze()).div(device_best_time.squeeze()).sum(axis=1)

    metric_sorted = stat.metric.sort_values()
    stat_sorted = stat.loc[metric_sorted.index]
    best_config = stat_sorted.index[0]
    best_config_index = stat.reset_index().loc[stat.reset_index()['index'] == best_config].dropna().index[0]

    res = dict(
        layer=stat.layer.iloc[0],
        best_config=best_config,
        best_metric=stat['metric'].min(),
        times=stat.loc[best_config, devices],
        times_order=[i.tolist().index(best_config_index) for i in stat[devices].T.to_numpy().argsort()],
        times_min=stat[devices].min()
    )

    if vis:
        visualize_metric(stat_sorted[devices], metric_sorted, name)

    if verbose:
        with open(osp.join(config_dir, name + '.json'), 'a') as ouf:
            ouf.write(json.dumps(stat.config[best_config]).replace('\\"', '"')[1:-1])
            ouf.write('\n')

        print(f'\tLayer {res["layer"]}, config={res["best_config"]}: metric={res["best_metric"]:.6f}, ' \
            f'times={[round(i, 6) for i in res["times"]]}, times_order={res["times_order"]}')
    
    return res


def visualize_metric(df: pd.DataFrame, metric: pd.Series, name: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.xlabel("metric", fontsize=14)
    plt.ylabel("time, ms", fontsize=14)
    plt.plot(df.reset_index(drop=True), label=df.columns)
    plt.plot(metric.reset_index(drop=True), '--', color='black', label='metric')
    plt.legend(prop={'size': 12})
    plt.xticks(range(len(metric)), [f"{m:.2f}" for m in metric])
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(6))
    tuner = name.split('.')[-1]
    if not osp.exists(osp.join(output_dir, tuner)):
        os.makedirs(osp.join(output_dir, tuner))
    plt.savefig(osp.join(output_dir, tuner, name + ".png"))
    plt.close()


@contextlib.contextmanager
def silence():
    sys.stdout, old = io.StringIO(), sys.stdout
    try:
        yield
    finally:
        sys.stdout = old
