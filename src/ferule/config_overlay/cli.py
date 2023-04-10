import os
import warnings
import typing as tp
from pathlib import Path

import click
import numpy as np

from . import HandleFile, Executor
from . import get_common_statistic, analyze_configs
from .. import target, target_host
from ..model_importer import ModelImporter


@click.group()
def cli1() -> None:
    pass


@click.command()
@click.argument('host_dev_logs', nargs=1, type=click.Path(exists=True, resolve_path=True), required=True)
@click.argument('target_dev_key', nargs=1, type=str, required=True)
@click.option('-m', '--model',  type=click.Choice(ModelImporter.available_models()), default=None)
@click.option('-t', '--type', 'dtype', type=click.Choice(['float16', 'float32']), default='float32', show_default=True,
              help='Specify whether the model should be run with single or half precision floating point values')
@click.option('-l', '--layer', 'layers', multiple=True, type=int, 
              help='The indices of the layers to be rendered. By default, the script collect statistics for all layers.')
@click.option('-r', '--rpc_tracker_host', 'host', envvar='TVM_TRACKER_HOST', help='RPC tracker host IP address')
@click.option('-p', '--rpc_tracker_port', 'port', type=int, envvar='TVM_TRACKER_PORT', help='RPC tracker host port')
@click.option('-T', '--target', default=target, show_default=True, help='Compilation target')
@click.option('-H', '--target_host', default=target_host, show_default=True, help='Compilation host target')
@click.option('-b', '--best', type=int, default=None, help='Amount of best layers that will be measured on other devices')
def collect(
    host_dev_logs: Path, 
    target_dev_key: str,
    model: tp.Optional[str], 
    dtype: str, 
    layers: tp.Sequence[str], 
    host: str, 
    port: int, 
    target: str, 
    target_host: str, 
    best: tp.Optional[int] = None,
    ) -> None:
    """Collect tuning statistics from the specified file on the target device.

    HOST_DEV_LOGS is a .json file obtained as a result of tuning.

    TARGET_DEV_KEY is the RPC key of the device on which the tuning will be performed.
    """
    handler = HandleFile(host_dev_logs, model, dtype)
    indices = layers if layers else list(range(len(handler)))
    executor = Executor(target, target_host, host, port, target_dev_key)
    for index, layers in enumerate(handler.layers):
        if index in indices:
            get_common_statistic(layers, executor, index, best)


@click.group()
def cli2() -> None:
    pass


@click.command()
@click.argument('logs', nargs=-1, type=click.Path(exists=True, resolve_path=True), required=True)
@click.option('-v', '--verbose', is_flag=True)
def analyze(logs: tp.Tuple[Path], verbose: bool) -> None:
    """Calculates the optimal common configuration for multiple devices.

    LOGS is .json configuration files, including those left after remeasurement using the "collect" method.
    """
    suffix = logs[0][logs[0].index("."):]
    assert np.array([suffix in log for log in logs]).all(), 'tuning statistics obtained with different configuration'

    host_to_target = dict()
    for log in logs:
        prefix = os.path.basename(log).split(".")[0]
        if '_' not in prefix:
            prefix = prefix + '_' + prefix
        target, host = prefix.split('_')
        if host in host_to_target:
            assert target not in host_to_target[host], f'files with prefix "{os.path.basename(log).split(".")[0]}" are duplicated'
            host_to_target[host][target] = HandleFile(log)
        else:
            host_to_target[host] = {target: HandleFile(log)}

    hosts = list(host_to_target.keys())
    assert np.array([len(host_to_target[hosts[0]].keys() ^ host_to_target[h].keys()) == 0 for h in hosts]).all(), \
    'for different hosts, a different set of target devices is specified'
    
    if len(host_to_target[hosts[0]]) != len(host_to_target):
        print('[INFO]: sets of hosts and targets are different')

    analyze_configs(host_to_target, verbose)


cli = click.CommandCollection(sources=[cli1, cli2])
cli1.add_command(collect)
cli2.add_command(analyze)


if __name__ == '__main__':
    cli()
