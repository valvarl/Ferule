import os
from pathlib import Path

from tvm import autotvm, auto_scheduler
from tvm.contrib import ndk

view_folder = os.path.join(str(Path.home()), 'ferule')
if not os.path.exists(view_folder):
    os.makedirs(view_folder)

log_dir = os.path.join(view_folder, "configs")
lib_dir = os.path.join(view_folder, "lib")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(lib_dir):
    os.makedirs(lib_dir)

target = 'llvm -model=snapdragon835 -mtriple=arm-linux-android -mattr=+neon'
target_host = 'llvm -mtriple=aarch64-linux-android-g++'


def autotvm_tuner_options(host: str, port: int, key: str, log_file: str):
    builder = autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15)
    options = {
        'n_trial': 512,
        'early_stopping': None,
        'measure_option': autotvm.measure_option(
            builder=builder,
            runner=autotvm.RPCRunner(
                key=key,
                host=host,
                port=port,
                number=50,
                repeat=10,
                timeout=15,
            ),
        ),
        'callbacks': [
            autotvm.callback.log_to_file(log_file),
        ],
    }
    return options


def ansor_tuner_options(host: str, port: int, key: str, log_file: str):
    builder = auto_scheduler.LocalBuilder(build_func=ndk.create_shared, timeout=15)
    options = auto_scheduler.TuningOptions(
        builder=builder,
        num_measure_trials=5000,
        num_measures_per_round=100,
        runner=auto_scheduler.RPCRunner(
            key=key,
            host=host,
            port=port,
        ),
        measure_callbacks=[
            auto_scheduler.RecordToFile(log_file),
        ],
    )
    return options


__all__ = ['target', 'target_host', 'autotvm_tuner_options', 'ansor_tuner_options', 'view_folder', 'log_dir', 'lib_dir']
