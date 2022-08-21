import os
import numpy as np

from tvm import relay, transform, autotvm, auto_scheduler
from tvm.contrib import ndk
from tvm.target import Target

from . import ansor_tuner_options, autotvm_tuner_options
from . import view_folder

log_dir = os.path.join(view_folder, "logs")
lib_dir = os.path.join(view_folder, "so")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(lib_dir):
    os.makedirs(lib_dir)


class Executor:
    def __init__(self, args):
        self.remote = None
        self.tracker = None
        self.args = args
        if 'tuner' in self.args:
            self.log_file = os.path.join(log_dir, f"{self.args['name']}.{self.args['tuner']}.json")
        self.target = Target(self.args['target'], host=self.args['target_host'])

    def _connect_tracker(self):
        from tvm import rpc

        print(
            "Tracker attempting connection on {}:{}".format(
                self.args['host'], self.args['port']
            )
        )
        self.tracker = rpc.connect_tracker(self.args['host'], self.args['port'])
        self.remote = self.tracker.request(
            self.args['key'], priority=0, session_timeout=None
        )
        print("Tracker connected to remote RPC server")

    def _disconnect_tracker(self):
        self.remote = None
        self.tracker = None

    def tune_autotvm(self, mod, params):
        from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
        tuner_options = autotvm_tuner_options(self.args, self.log_file)

        print("Extract autotvm tasks...")
        tasks = autotvm.task.extract_from_program(
            mod["main"], target=self.args['target'], target_host=self.args['target_host'], params=params)
        
        for idx, task in enumerate(tasks):
            print("========== Task %d ==========" % (idx))
            print(task)

        print("Begin tuning...")
        print("Tuning logs are available at %s" % self.log_file)
        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

            # create tuner
            tuner = self.args['method']
            if tuner == "xgb" or tuner == "xgb-rank":
                tuner_obj = XGBTuner(task, loss_type="rank")
            elif tuner == "ga":
                tuner_obj = GATuner(task, pop_size=50)
            elif tuner == "random":
                tuner_obj = RandomTuner(task)
            elif tuner == "gridsearch":
                tuner_obj = GridSearchTuner(task)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            # do tuning
            tuner_options['n_trial'] = min(tuner_options['n_trial'], len(task.config_space))
            tuner_options['callbacks'].append(autotvm.callback.progress_bar(tuner_options['n_trial'], prefix=prefix))
            
            tuner_obj.tune(**tuner_options)
            tuner_options['callbacks'].pop()

    def compile_autotvm(self, mod, params, log_file=None):
        if log_file is None:
            log_file = self.log_file
        with autotvm.apply_history_best(log_file):
            print("Compile...")
            with transform.PassContext(opt_level=3):
                lib = relay.build_module.build(
                    mod, target=self.target, params=params)
        lib_path = os.path.join(lib_dir, f"{self.args['name']}.atvm.so")
        print("Source object was compiled at %s" % lib_path)
        lib.export_library(lib_path, ndk.create_shared)

    def tune_ansor(self, mod, params):
        print("Extract ansor tasks...")
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod["main"], params, target=self.target)
        
        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" %
                  (idx, task.workload_key))
            print(task.compute_dag)

        print("Begin tuning...")
        print("Tuning logs are available at %s" % self.log_file)
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tuner.tune(ansor_tuner_options(self.args, self.log_file))

    def compile_ansor(self, mod, params, log_file=None):
        if log_file is None:
            log_file = self.log_file
        with auto_scheduler.ApplyHistoryBest(log_file):
            print("Compile...")
            with transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=self.target, params=params)
        lib_path = os.path.join(lib_dir, f"{self.args['name']}.ansor.so")
        print("Source object was compiled at %s" % lib_path)
        lib.export_library(lib_path, ndk.create_shared)

    def benchmark(self, input_path=None):
        from tvm.contrib import graph_executor

        if self.remote is None:
            self._connect_tracker()

        if input_path is None:
            input_path = os.path.join(lib_dir, f"{self.args['name']}.{self.args['tuner']}.so")

        print("Uploading binary...")
        self.remote.upload(input_path)
        lib = self.remote.load_module(os.path.basename(input_path))
        ctx = self.remote.cpu(0)
        m = graph_executor.GraphModule(lib["default"](ctx))

        print("Starting measurements...")
        ftimer = m.module.time_evaluator("run", ctx, repeat=10, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res))
        )

        self._disconnect_tracker()
