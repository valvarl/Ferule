import os
import warnings
import typing as tp

import click

from . import HandleFile, Executor
from . import draw_config_overlay_graph, mesure_and_draw_co_graph
from .. import target, target_host
from ..model_importer import ModelImporter

@click.command()
@click.argument('logs', nargs=-1, type=click.Path(exists=True, resolve_path=True), required=True)
@click.option('-m', '--model',  type=click.Choice(ModelImporter.available_models()), default=None)
@click.option('-t', '--type', 'dtype', type=click.Choice(['float16', 'float32']), default='float32', show_default=True,
              help='Specify whether the model should be run with single or half precision floating point values')
@click.option('-l', '--layer', 'layers', multiple=True, type=int, 
help='The indices of the layers to be rendered. By default, the script builds graphs for all layers.')
@click.option('-r', '--rpc_tracker_host', 'host', envvar='TVM_TRACKER_HOST', help='RPC tracker host IP address')
@click.option('-p', '--rpc_tracker_port', 'port', type=int, envvar='TVM_TRACKER_PORT', help='RPC tracker host port')
@click.option('-k', '--rpc_key', 'keys',  multiple=True, show_default=True, 
help='List of RPC keys for which execution time will be measured')
@click.option('-T', '--target', default=target, show_default=True, help='Compilation target')
@click.option('-H', '--target_host', default=target_host, show_default=True, help='Compilation host target')
def cli(logs, model: tp.Optional[str], dtype: str, layers: tp.Sequence[str], 
    host: str, port: int, keys: tp.Sequence[str], target: str, target_host: str):
    """Takes information about the network configuration from log file(s) and builds a comparative graph.

    LOGS is a list of log files.

    Note: Don't specify RPC keys unless you plan to measure execution time. Otherwise, the measurement results will be taken from the logs.
    """
    handlers = [HandleFile(log, model, dtype) for log in logs]
    indices = layers if layers else list(range(len(handlers[0])))
    if not keys:
        records = [len(handler) for handler in handlers]
        if len(set(records)) != 1:
            warnings.warn("The logs contain a different number of entries. If a conflict occurs, the last entry will be used.")
        for index, layer in enumerate(handlers[0].layers):
            if index in indices:
                same_layer = []
                for handler in handlers:
                    same_layer.append(handler[layer])
                if len(set([w.tuner for w in same_layer])) != 1:
                    raise RuntimeError("Found logs of different autotuners.")
                if len(set([len(w.configs) for w in same_layer])) != 1:
                    warnings.warn("Layer %d contains a different number of trials in log files." % index)
                    
                labels = [os.path.basename(handler.file).split('.')[0] for handler in handlers]
                draw_config_overlay_graph(same_layer, labels, index)
    
    elif len(logs) == 1:
        executors = [Executor(target, target_host, host, port, key) for key in keys]
        for index, layer in enumerate(handlers[0].layers):
            if index in indices:
                mesure_and_draw_co_graph(layer, executors, index)

    else:
        executors = [Executor(target, target_host, host, port, key) for key in keys]
        for index, layers in enumerate(zip(*[handler.layers for handler in handlers])):
            if index in indices:
                mesure_and_draw_co_graph(layers, executors, index)


if __name__ == '__main__':
    cli()
