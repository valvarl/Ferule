import os
import warnings
import typing as tp
from pathlib import Path

import click
import pandas as pd

from . import HandleFile, Executor
from . import get_common_statistic, calculate_metric
from .. import target, target_host
from ..model_importer import ModelImporter


@click.group()
def cli1() -> None:
    pass


@click.command()
@click.argument('logs', nargs=1, type=click.Path(exists=True, resolve_path=True), required=True)
@click.option('-m', '--model',  type=click.Choice(ModelImporter.available_models()), default=None)
@click.option('-t', '--type', 'dtype', type=click.Choice(['float16', 'float32']), default='float32', show_default=True,
              help='Specify whether the model should be run with single or half precision floating point values')
@click.option('-l', '--layer', 'layers', multiple=True, type=int, 
              help='The indices of the layers to be rendered. By default, the script collect statistics for all layers.')
@click.option('-r', '--rpc_tracker_host', 'host', envvar='TVM_TRACKER_HOST', help='RPC tracker host IP address')
@click.option('-p', '--rpc_tracker_port', 'port', type=int, envvar='TVM_TRACKER_PORT', help='RPC tracker host port')
@click.option('-k', '--rpc_key', 'keys',  multiple=True, show_default=True, 
              help='List of RPC keys for which execution time will be measured')
@click.option('-T', '--target', default=target, show_default=True, help='Compilation target')
@click.option('-H', '--target_host', default=target_host, show_default=True, help='Compilation host target')
@click.option('-b', '--best', type=int, default=None, help='Amount of best layers that will be measured on other devices')
@click.option('-e', '--estimate', default=None, type=click.Path(exists=True, resolve_path=True),
              help='Uses previous information (.csv) about shared tuning to evaluate the success of choosing a common config for a layer.')
def collect(
    logs: Path, 
    model: tp.Optional[str], 
    dtype: str, 
    layers: tp.Sequence[str], 
    host: str, 
    port: int, 
    keys: tp.Sequence[str], 
    target: str, 
    target_host: str, 
    best: tp.Optional[int] = None,
    estimate: tp.Optional[Path] = None
    ) -> None:
    """Collect inference statistics on provided devices by shared config.

    LOGS is .json file obtained as a result of tuning.
    """
    handler = HandleFile(logs, model, dtype)
    indices = layers if layers else list(range(len(handler)))
    executors = [Executor(target, target_host, host, port, key) for key in keys]
    for index, layers in enumerate(handler.layers):
        if index in indices:
            get_common_statistic(layers, executors, index, best, estimate)


@click.group()
def cli2() -> None:
    pass


@click.command()
@click.argument('logs', nargs=1, type=click.Path(exists=True, resolve_path=True), required=True)
@click.option('--vis', is_flag=True, default=False, help='Visualize')
def common(
    logs: Path,
    vis: bool = False
    ) -> None:
    """Generates a configuration file based on the metric results for each layer setting.

    LOGS is .csv file obtained as a result of joint tuning.
    """
    data = pd.read_csv(logs)
    name = os.path.basename(logs).replace('.csv', '')
    for layer in data.layer.unique():
        calculate_metric(data[data.layer == layer], name, vis, verbose=True)


cli = click.CommandCollection(sources=[cli1, cli2])
cli1.add_command(collect)
cli2.add_command(common)


if __name__ == '__main__':
    cli()
