import os
import typing as tp
import numpy as np

import tvm
from tvm import relay, transform
from tvm import autotvm, auto_scheduler
from tvm.contrib import ndk
from tvm.target import Target

from . import target, target_host, lib_dir
from . import ansor_tuner_options, autotvm_tuner_options


class Executor:
    def __init__(
        self, 
        target: str = target, target_host: str = target_host,
        host: str = "0.0.0.0", port: int = 9190, key: str = "android",
        tuner: tp.Optional[str] = None, method: str = "gridsearch") -> None:
        self.remote = None
        self.tracker = None
        self.target = Target(target, host=target_host)
        self.host, self.port, self.key = host, port, key
        self.tuner, self.method = tuner, method

    def _connect_tracker(self):
        from tvm import rpc

        print(
            "Tracker attempting connection on {}:{}".format(
                self.host, self.port
            )
        )
        self.tracker = rpc.connect_tracker(self.host, self.port)
        self.remote = self.tracker.request(
            self.key, priority=0, session_timeout=None
        )
        print("Tracker connected to remote RPC server")

    def _disconnect_tracker(self):
        self.remote = None
        self.tracker = None

    def tune_autotvm(self, mod, params, log_file):
        from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
        tuner_options = autotvm_tuner_options(self.host, self.port, self.key, log_file)

        print("Extract autotvm tasks...")
        tasks = autotvm.task.extract_from_program(
            mod["main"], target=self.target, params=params)
        
        for idx, task in enumerate(tasks):
            print("========== Task %d ==========" % (idx))
            print(task)

        print("Begin tuning...")
        print("Tuning logs are available at %s" % log_file)
        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

            # create tuner
            if self.method == "xgb" or self.method == "xgb-rank":
                tuner_obj = XGBTuner(task, loss_type="rank")
            elif self.method == "ga":
                tuner_obj = GATuner(task, pop_size=50)
            elif self.method == "random":
                tuner_obj = RandomTuner(task)
            elif self.method == "gridsearch":
                tuner_obj = GridSearchTuner(task)
            else:
                raise ValueError("Invalid tuner: " + self.method)

            # do tuning
            tuner_options['n_trial'] = min(tuner_options['n_trial'], len(task.config_space))
            tuner_options['callbacks'].append(autotvm.callback.progress_bar(tuner_options['n_trial'], prefix=prefix))
            
            tuner_obj.tune(**tuner_options)
            tuner_options['callbacks'].pop()

    def compile_autotvm(self, mod, params, log_file: str, lib_dir: str = lib_dir):
        with autotvm.apply_history_best(log_file):
            print("Compile...")
            with transform.PassContext(opt_level=3):
                if params is not None:
                    lib = relay.build_module.build(mod, target=self.target, params=params)
                else:  # compile one task layer
                    lib = tvm.build(mod, target=self.target)
        self.lib_path = os.path.join(lib_dir, os.path.basename(log_file).rstrip('.json ') + '.so')
        print("Source object was compiled at %s" % self.lib_path)
        lib.export_library(self.lib_path, ndk.create_shared)

    def tune_ansor(self, mod, params, log_file: str):
        print("Extract ansor tasks...")
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod["main"], params, target=self.target)
        
        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" %
                  (idx, task.workload_key))
            print(task.compute_dag)

        print("Begin tuning...")
        print("Tuning logs are available at %s" % log_file)
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tuner.tune(ansor_tuner_options(self.host, self.port, self.key, log_file))

    def compile_ansor(self, mod, params, log_file: str, lib_dir: str = lib_dir):
        with auto_scheduler.ApplyHistoryBest(log_file):
            print("Compile...")
            with transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                if params is not None:  # compile whole model
                    lib = relay.build(mod, target=self.target, params=params)
                else:  # compile one task layer
                    lib = tvm.build(mod, target=self.target)
        self.lib_path = os.path.join(lib_dir, os.path.basename(log_file).rstrip('.json ') + '.so')
        print("Source object was compiled at %s" % self.lib_path)
        lib.export_library(self.lib_path, ndk.create_shared)

    def benchmark(self, input_path=None) -> float:
        from tvm.contrib import graph_executor

        if self.remote is None:
            self._connect_tracker()

        if input_path is None:
            input_path = self.lib_path

        print("Uploading binary...")
        self.remote.upload(input_path)
        lib = self.remote.load_module(os.path.basename(input_path))
        ctx = self.remote.cpu(0)
        m = graph_executor.GraphModule(lib["default"](ctx))

        print("Starting measurements...")
        ftimer = m.module.time_evaluator("run", ctx, repeat=10, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        mean_res, std_res = np.mean(prof_res), np.std(prof_res)
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (mean_res, std_res))
        self._disconnect_tracker()
        return np.mean(prof_res)

    def xbenchmark(self, args: tp.Sequence[tvm.te.tensor.Tensor], dtype: str = "float32", input_path=None) -> float:
        if self.remote is None:
            self._connect_tracker()

        if input_path is None:
            input_path = self.lib_path

        print("Uploading binary...")
        self.remote.upload(input_path)
        lib = self.remote.load_module(os.path.basename(input_path))
        ctx = self.remote.cpu(0)

        inputs = []
        for tensor in args:
            shape = [int(j) for j in tensor.shape]
            inputs.append(tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx))

        time_f = lib.time_evaluator(lib.entry_name, ctx, number=10)
        prof_res = np.array(time_f(*inputs).results) * 1e3  # convert to millisecond
        mean_res, std_res = np.mean(prof_res), np.std(prof_res)
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (mean_res, std_res))
        self._disconnect_tracker()
        return np.mean(prof_res)
