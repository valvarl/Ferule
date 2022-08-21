# Ferule
Ferule is a pun that explains the role of this library well. The word for rigor in education "motivating ruler" brings together tools for measuring and comparing performance of neural networks.

A feature of the project is that it uses the functionality of the [TVM](https://github.com/apache/tvm) engine. The library uses an RPC connection to deliver data and take measurements on a mobile device. The following tools are currently available:
* **cross-compile** allows you to tune and get the source object for one device and measure performance of the compiled network on the other;
* **config-overlay** takes the logs left after tuning and builds training graphs, where the x-axis is the index of the trials, and the y-axis is the time of the layer inference.

## Quickstart
Run the following code inside the repository
```console
foo@bar:~$ python -m pip install --upgrade pip build setuptools
foo@bar:~$ python -m build --sdist --wheel
foo@bar:~$ python -m pip install . --prefer-binary --force-reinstall --find-links dist/
```
Tools are now available from the command line
```console
foo@bar:~$ cross-compile
Usage: cross-compile [OPTIONS] COMMAND [ARGS]...
```

## cross-compile
Program for cross-compilation and collection inference statistics. It works in two modes: tune and execute. 

`~$ cross-compile tune` tunes the neural network according to the selected autotuner, compiles the model and measures the run time. Detailed information can be obtained using the `--help` flag.
```console
foo@bar:~$ cross-compile tune atvm mace_mobilenet_v1 -p 9090 -k sd888
```
`~$ cross-compile exec` measure the execution time of source object on the passed device. It can accept JSON obtained as a result of tuning, while additionally compiling 
(specify `--target` and `--target_host` if needed).
```console
foo@bar:~$ cross-compile exec atvm sd888.mace_mobilenet_v1.float32.atvm.so -p 9090 -k kirin710
```
Additionally, there is `~$ cross-compile view` command that displays the path to the folder where models, logs and source objects are saved.

## config-overlay
This command allows you to visually evaluate and compare the quality of tuning on each layer.
```console
foo@bar:~$ config-overlay `cross-compile view`/logs/*mobilenet_v1.float32.atvm.json
```
![output](img/6.conv2d_NCHWc.x86.%5B1%2C128%2C56%2C56%5D.png)
On the chart, the indices are sorted by the inference time of the first specified layer. It is easy to see that after sorting one of the graphs, the rest also show growth.