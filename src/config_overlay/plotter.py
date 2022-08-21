import os
import json
import typing as tp

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .dispatcher import Layer, Tuner

output_dir = "output"


def draw_config_overlay_graph(layers: tp.List[Layer], index: int, labels: tp.List[str]) -> None:
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
    
    df = pd.DataFrame(data, columns=range(len(layers)), index=range(len(layers[0].configs))).sort_values(0) * 1000
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
