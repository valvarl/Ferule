import os
import inspect

from tvm import relay

from .. import view_folder

class ModelImporter:
    @classmethod
    def available_models(cls):
        models = []
        for method in inspect.getmembers(cls):
            if "import_" in method[0]:
                models.append(method[0].split("import_")[1])
        return models

    def __call__(self, model, *args, **kwargs):
        for method in inspect.getmembers(type(self)):
            if "import_" + model == method[0]:
                return method[1](self, *args, **kwargs)
        raise ValueError("import_" + model + " not found.")


    def get_onnx_from_tf1(self, model_url, filename, input_names, output_names, shape_override = None):
        tf_model_file = os.path.join(view_folder, "models", "%s.pb" % filename)

        print("Collecting model...")
        from tvm.contrib import download
        download.download(model_url, tf_model_file)
        # converted using command line:
        # python -m tf2onnx.convert --graphdef mace_resnet-v2-50.pb --output mace_resnet-v2-50.onnx --inputs input:0[1,224,224,3] --outputs resnet_v2_50/predictions/Reshape_1:0
        onnx_model_file = os.path.join(view_folder, "models", "%s.onnx" % filename)
        if os.path.exists(onnx_model_file) == False:
            import tf2onnx
            import tensorflow as tf
            try:
                tf_compat_v1 = tf.compat.v1
            except ImportError:
                tf_compat_v1 = tf
            # Tensorflow utility functions
            import tvm.relay.testing.tf as tf_testing

            with tf_compat_v1.gfile.GFile(tf_model_file, "rb") as f:
                graph_def = tf_compat_v1.GraphDef()
                graph_def.ParseFromString(f.read())
                #graph = tf.import_graph_def(graph_def, name="")
                # Call the utility to import the graph definition into default graph.
                graph_def = tf_testing.ProcessGraphDefParam(graph_def)

                model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(graph_def,
                    name=filename, input_names=input_names, output_names=output_names,
                    shape_override = shape_override,
                    output_path=onnx_model_file)

        return onnx_model_file


    def get_graphdef_from_tf1(self, model_url, filename):
        graph_def = None
        tf_model_file = os.path.join(view_folder, "models", "%s.pb" % filename)
        
        print("Collecting model...")
        from tvm.contrib import download
        download.download(model_url, tf_model_file)
        # converted using command line:
        # python -m tf2onnx.convert --graphdef mace_resnet-v2-50.pb --output mace_resnet-v2-50.onnx --inputs input:0[1,224,224,3] --outputs resnet_v2_50/predictions/Reshape_1:0
        onnx_model_file = os.path.join(view_folder, "models", "%s.onnx" % filename)
        import tensorflow as tf
        try:
            tf_compat_v1 = tf.compat.v1
        except ImportError:
            tf_compat_v1 = tf
        # Tensorflow utility functions
        import tvm.relay.testing.tf as tf_testing

        with tf_compat_v1.gfile.GFile(tf_model_file, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        return graph_def


    def import_mace_mobilenet_v1(self, tuner, dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/mobilenet-v1/mobilenet-v1-1.0.pb"
        filename = "mace_mobilenet-v1-1.0"
        input_names = ["input:0"]
        output_names = ["MobilenetV1/Predictions/Reshape_1:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names)
        import onnx
        model = onnx.load(onnx_model_file)
        shape_dict = {'input:0': [1, 224, 224, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return mod, params


    def import_mace_resnet50_v2(self, tuner, dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/resnet-v2-50/resnet-v2-50.pb"
        filename = "mace_resnet-v2-50"
        input_names = ["input:0"]
        shape_override = {"input:0": [1, 299, 299, 3]}
        output_names = ["resnet_v2_50/predictions/Reshape_1:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names, shape_override)
        import onnx
        model = onnx.load(onnx_model_file)
        mod, params = relay.frontend.from_onnx(model, shape_override, freeze_params=True)
        # DEELVIN-207
        # mod = relay.transform.InferType()(mod)
        # mod = relay.transform.ToMixedPrecision()(mod)
        # print(mod)

        mod = relay.quantize.prerequisite_optimize(mod, params)

        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return mod, params


    def import_mace_inception_v3(self, tuner, dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/inception-v3/inception-v3.pb"
        filename = "mace_inception-v3"
        input_names = ["input:0"]
        output_names = ["InceptionV3/Predictions/Reshape_1:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names)
        import onnx
        model = onnx.load(onnx_model_file)
        shape_dict = {'input:0': [1, 299, 299, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        
        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return mod, params


    def import_mace_yolo_v3(self, tuner, dtype="float32"):
        model_url = "http://cnbj1.fds.api.xiaomi.com/mace/miai-models/yolo-v3/yolo-v3.pb"
        filename = "mace_yolo-v3"
        graph_def = self.get_graphdef_from_tf1(model_url, filename)
        shape_dict = {"input_1": (1, 416, 416, 3)}
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict,
                                        outputs=["conv2d_59/BiasAdd","conv2d_67/BiasAdd","conv2d_75/BiasAdd"])

        if tuner == 'atvm':
            # We assume our model's heavily-layout sensitive operators only consist of nn.conv2d
            desired_layouts = {'nn.conv2d': ['NCHW', 'default']}

            # Convert the layout to NCHW
            # RemoveUnunsedFunctions is used to clean up the graph.
            from tvm.relay import transform
            seq = transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                            relay.transform.ConvertLayout(desired_layouts)])
            with transform.PassContext(opt_level=3):
                mod = seq(mod)

        mod = relay.quantize.prerequisite_optimize(mod, params)

        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
            mod = relay.quantize.prerequisite_optimize(mod, params)
        return mod, params

   
def downcast_fp16(func, module):
    from tvm.relay.expr_functor import ExprMutator
    from tvm.relay.expr import Call, Var, Constant, TupleGetItem
    from tvm.relay import transform as _transform
    from tvm.relay import cast
    from tvm.ir import IRModule
    from tvm.relay import function as _function

    """Downcast to fp16 mutator
    Parameters
    ---------
    graph: Function
        The original graph.
    Retruns
    -------
    The graph after dowmcasting to half-precision floating-point.
    """
    filter_list = ["vision.get_valid_counts", "vision.non_max_suppression"]

    class DowncastMutator(ExprMutator):
        """Downcast to fp16 mutator"""

        def visit_call(self, call):
            dtype = "float32" if call.op.name in filter_list else "float16"
            new_fn = self.visit(call.op)
            # Collect the original dtypes
            type_list = []
            if call.op.name in filter_list:
                # For NMS
                for arg in call.args:
                    if isinstance(arg, TupleGetItem) and isinstance(
                        arg.tuple_value, Call
                    ):
                        tuple_types = arg.tuple_value.checked_type.fields
                        type_list.append(tuple_types[arg.index].dtype)
                if call.op.name == "vision.get_valid_counts":
                    tuple_types = call.checked_type.fields
                    for cur_type in tuple_types:
                        type_list.append(cur_type.dtype)

            args = [self.visit(arg) for arg in call.args]
            new_args = list()
            arg_idx = 0
            for arg in args:
                if isinstance(arg, (Var, Constant)):
                    new_args.append(cast(arg, dtype=dtype))
                else:
                    if call.op.name in filter_list:
                        if (
                            isinstance(arg, TupleGetItem)
                            and type_list[arg_idx] == "int32"
                        ):
                            new_args.append(arg)
                        else:
                            new_args.append(cast(arg, dtype=dtype))
                    else:
                        new_args.append(arg)
                arg_idx += 1
            if (
                call.op.name in filter_list
                and call.op.name != "vision.get_valid_counts"
            ):
                return cast(Call(new_fn, new_args, call.attrs), dtype="float16")
            return Call(new_fn, new_args, call.attrs)

    class UpcastMutator(ExprMutator):
        """upcast output back to fp32 mutator"""

        def visit_call(self, call):
            return cast(call, dtype="float32")

    def infer_type(node, mod=None):
        """A method to infer the type of an intermediate node in the relay graph."""
        if isinstance(mod, IRModule):
            mod["main"] = _function.Function(relay.analysis.free_vars(node), node)
            mod = _transform.InferType()(mod)
            entry = mod["main"]
            ret = entry.body
        else:
            new_mod = IRModule.from_expr(node)
            if mod is not None:
                new_mod.update(mod)
                new_mod = _transform.InferType()(new_mod)
                entry = new_mod["main"]
                ret = entry if isinstance(node, _function.Function) else entry.body

        return ret

    func = infer_type(func, module)
    downcast_pass = DowncastMutator()
    func = downcast_pass.visit(func)
    upcast_pass = UpcastMutator()
    func = upcast_pass.visit(func)
    func = infer_type(func, module)
    new_mod = IRModule.from_expr(func)
    # new_mod.update(module)
    return new_mod
