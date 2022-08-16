import os

from tvm import autotvm, auto_scheduler
from tvm.contrib import ndk

view_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'view')
if not os.path.exists(view_folder):
    os.makedirs(view_folder)

target = 'llvm -model=snapdragon835 -mtriple=arm-linux-android -mattr=+neon'
target_host = 'llvm -mtriple=aarch64-linux-android-g++'


def autotvm_tuner_options(args, log_file):
    builder = autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15)
    options = {
        'n_trial': 150,
        'early_stopping': None,
        'measure_option': autotvm.measure_option(
            builder=builder,
            runner=autotvm.RPCRunner(
                key=args['key'],
                host=args['host'],
                port=args['port'],
                number=50,
                timeout=15,
            ),
        ),
        'callbacks': [
            autotvm.callback.log_to_file(log_file),
        ],
    }
    return options


def ansor_tuner_options(args, log_file):
    builder = auto_scheduler.LocalBuilder(build_func=ndk.create_shared, timeout=15)
    options = auto_scheduler.TuningOptions(
        builder=builder,
        num_measure_trials=5000,
        num_measures_per_round=100,
        runner=auto_scheduler.RPCRunner(
            key=args['key'],
            host=args['host'],
            port=args['port'],
        ),
        measure_callbacks=[
            auto_scheduler.RecordToFile(log_file),
        ],
    )
    return options


__all__ = ['target', 'target_host', 'autotvm_tuner_options', 'ansor_tuner_options', 'view_folder']
