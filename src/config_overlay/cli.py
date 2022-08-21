import os
import warnings

import click

from . import HandleFile
from . import draw_config_overlay_graph

@click.command()
@click.argument('logs', nargs=-1, type=click.Path(exists=True, resolve_path=True), required=True)
def cli(logs):
    """Takes information about the network configuration from different logs and builds a comparative graph.

    LOGS is a list of log files. The graph to be built will be sorted by the indexes of the first specified file.
    """
    handlers = [HandleFile(log) for log in logs]
    records = [len(handler) for handler in handlers]
    if len(set(records)) != 1:
        warnings.warn("The logs contain a different number of entries. If a conflict occurs, the last entry will be used.")
    for index, layer in enumerate(handlers[0].layers):
        # print(layer.configs[0]['i'][0])
        same_layer = []
        for handler in handlers:
            same_layer.append(handler[layer])
        if len(set([w.tuner for w in same_layer])) != 1:
            raise RuntimeError("Found logs of different autotuners.")
        if len(set([len(w.configs) for w in same_layer])) != 1:
            warnings.warn("Layer %d contains a different number of trials in log files." % index)
            
        labels = [os.path.basename(handler.file).split('.')[0] for handler in handlers]
        draw_config_overlay_graph(same_layer, index, labels)

if __name__ == '__main__':
    cli()
