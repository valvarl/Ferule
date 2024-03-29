import os
import typing as tp

import click

from ..model_importer import ModelImporter
from ..executor import Executor
from .. import target, target_host, log_dir

@click.group()
def cli1() -> None:
    pass


@click.command()
@click.argument('tuner', type=click.Choice(['atvm', 'ansor']))
@click.argument('model', type=click.Choice(ModelImporter.available_models()))
@click.option('-t', '--type', 'dtype',type=click.Choice(['float16', 'float32']), default='float32', show_default=True,
              help='Specify whether the model should be run with single or half precision floating point values')
@click.option('-m', '--method', type=click.Choice(['gridsearch', 'xgb', 'ga', 'random']),
              default='gridsearch', show_default=True, help='AutoTVM tuning method')
@click.option('-r', '--rpc_tracker_host', 'host', envvar='TVM_TRACKER_HOST', help='RPC tracker host IP address')
@click.option('-p', '--rpc_tracker_port', 'port', type=int, envvar='TVM_TRACKER_PORT', help='RPC tracker host port')
@click.option('-k', '--rpc_key', 'key', default='android', show_default=True, help='RPC key to use')
@click.option('-T', '--target', default=target, show_default=True, help='Compilation target')
@click.option('-H', '--target_host', default=target_host, show_default=True, help='Compilation host target')
def tune(tuner: str, model: str, dtype: str, method: str, host: str, port: int, key: str, target: str, target_host: str) -> None:
    """Tune model using AutoTVM/Auto-Scheduler and mesure performance."""
    importer = ModelImporter()
    mod, params = importer(model, tuner=tuner, dtype=dtype)
    log_file = os.path.join(log_dir, '.'.join([key, model, dtype, tuner, 'json']))

    executor = Executor(target, target_host, host, port, key, tuner, method)
    if tuner == 'atvm':
        executor.tune_autotvm(mod, params, log_file)
        executor.compile_autotvm(mod, params, log_file)
    elif tuner == 'ansor':
        executor.tune_ansor(mod, params, log_file)
        executor.compile_ansor(mod, params, log_file)

    executor.benchmark()


@click.group()
def cli2() -> None:
    pass


@click.command()
@click.argument('input_path', type=click.Path(exists=True, resolve_path=True))
@click.option('-r', '--rpc_tracker_host', 'host', envvar='TVM_TRACKER_HOST', help='RPC tracker host IP address')
@click.option('-p', '--rpc_tracker_port', 'port', type=int, envvar='TVM_TRACKER_PORT', help='RPC tracker host port')
@click.option('-k', '--rpc_key', 'key', default='android', show_default=True, help='RPC key to use')
@click.option('-T', '--target', default=target, show_default=True, help='Compilation target')
@click.option('-H', '--target_host', default=target_host, show_default=True, help='Compilation host target')
def exec(input_path, host: str, port: int, key: str, target: str, target_host: str) -> None:
    '''Execute model on selected device and takes measurements. The source object must be specified as the
    model. JSON also possible - it will be compiled before running, itc specify target/target_host. '''
    executor = Executor(target, target_host, host, port, key)
    if input_path.endswith('.json'):
        input_file = os.path.basename(input_path).split('.')
        _, model, dtype, tuner, _ = input_file

        importer = ModelImporter()
        mod, params = importer(model, tuner=tuner, dtype=dtype)
        if input_path.endswith('.atvm.json'):
            executor.compile_autotvm(mod, params, input_path)
        elif input_path.endswith('.ansor.json'):
            executor.compile_ansor(mod, params, input_path)
        executor.benchmark()

    elif input_path.endswith(".so"):
        executor.benchmark(input_path)
    else:
        raise NameError('Only source object/JSON file accepted.')


@click.group()
def cli3():
    pass


@click.command()
def view():
    """Inspect working directory."""
    from .. import view_folder
    click.echo(view_folder)


cli = click.CommandCollection(sources=[cli1, cli2, cli3])
cli1.add_command(tune)
cli2.add_command(exec)
cli3.add_command(view)

if __name__ == '__main__':
    cli()
