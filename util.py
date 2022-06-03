import os

import numpy as np
import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_executor as runtime


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
    """Get the symbol definition and random weight of a network"""

    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if 'split' in name:
        Nsplit = int( name.split('split-')[1].split('_')[0] )
        if 'mobilenet' in name:
            if Nsplit not in [4, 5, 6, 23, 27]: 
                print(bcolors.FAIL + '\t mobilenet_split-N options: 4, 5, 6, 23, 27'.format(Nsplit) + bcolors.ENDC); import sys; sys.exit(0)
        if 'squeezenet' in name:
            if Nsplit not in [6, 10, 24, 32, 36, 37, 38]: 
                print(bcolors.FAIL + '\t squeezenet_split-N options: 6, 10, 24, 32, 36, 37, 38'.format(Nsplit) + bcolors.ENDC); import sys; sys.exit(0)
        if 'inception_v3' in name:
            if Nsplit not in [5, 88, 107]: 
                print(bcolors.FAIL + '\t inception_v3_split-N options: 5, 88, 107'.format(Nsplit) + bcolors.ENDC); import sys; sys.exit(0)
        if 'vgg-19' in name:
            if Nsplit not in [20]: 
                print(bcolors.FAIL + '\t vgg-19_split-N options: 20'.format(Nsplit) + bcolors.ENDC); import sys; sys.exit(0)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )

    elif name == "mobilenet_split-4_0":
        mod, params = relay.testing.mobilenet_split.get_workload(N=4,npartition=0, batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size,64,56,56);
    elif name == "mobilenet_split-4_1":
        mod, params = relay.testing.mobilenet_split.get_workload(N=4,npartition=1, batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size,64,56,56);

    elif name == "mobilenet_split-5_0":
        mod, params = relay.testing.mobilenet_split.get_workload(N=5,npartition=0, batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size,128,56,56);
    elif name == "mobilenet_split-5_1":
        mod, params = relay.testing.mobilenet_split.get_workload(N=5,npartition=1, batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size,128,56,56);

    elif name == "mobilenet_split-6_0":
        mod, params = relay.testing.mobilenet_split.get_workload(N=6,npartition=0, batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size,128,56,56);
    elif name == "mobilenet_split-6_1":
        mod, params = relay.testing.mobilenet_split.get_workload(N=6,npartition=1, batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size,128,56,56);

    elif name == "mobilenet_split-23_0":
        mod, params = relay.testing.mobilenet_split.get_workload(N=23,npartition=0, batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size,512,14,14); 
    elif name == "mobilenet_split-23_1":
        mod, params = relay.testing.mobilenet_split.get_workload(N=23,npartition=1, batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size,512,14,14);

    elif name == "mobilenet_split-25_0":
        mod, params = relay.testing.mobilenet_split.get_workload(N=25,npartition=0, batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size,1024,7,7);
    elif name == "mobilenet_split-25_1":
        mod, params = relay.testing.mobilenet_split.get_workload(N=25,npartition=1, batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size,1024,7,7);

    elif name == "mobilenet_split-27_0":
        mod, params = relay.testing.mobilenet_split.get_workload(N=27,npartition=0, batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size,1024,7,7);
    elif name == "mobilenet_split-27_1":
        mod, params = relay.testing.mobilenet_split.get_workload(N=27,npartition=1, batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size,1024,7,7);


    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "squeezenet_split-6_0":
        mod, params = relay.testing.squeezenet_split.get_workload(N=6, npartition=0, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size, 128, 55, 55)
    elif name == "squeezenet_split-6_1":
        mod, params = relay.testing.squeezenet_split.get_workload(N=6, npartition=1, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size, 128, 55, 55)

    elif name == "squeezenet_split-10_0":
        mod, params = relay.testing.squeezenet_split.get_workload(N=10, npartition=0, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size, 128, 27, 27)
    elif name == "squeezenet_split-10_1":
        mod, params = relay.testing.squeezenet_split.get_workload(N=10, npartition=1, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size, 128, 27, 27)

    elif name == "squeezenet_split-24_0":
        mod, params = relay.testing.squeezenet_split.get_workload(N=24, npartition=0, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size, 384, 13, 13)
    elif name == "squeezenet_split-24_1":
        mod, params = relay.testing.squeezenet_split.get_workload(N=24, npartition=1, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size, 384, 13, 13)

    elif name == "squeezenet_split-32_0":
        mod, params = relay.testing.squeezenet_split.get_workload(N=32, npartition=0, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size, 512, 13, 13)
    elif name == "squeezenet_split-32_1":
        mod, params = relay.testing.squeezenet_split.get_workload(N=32, npartition=1, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size, 512, 13, 13)

    elif name == "squeezenet_split-36_0":
        mod, params = relay.testing.squeezenet_split.get_workload(N=36, npartition=0, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size, 1000, 13, 13)
    elif name == "squeezenet_split-36_1":
        mod, params = relay.testing.squeezenet_split.get_workload(N=36, npartition=1, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size, 1000, 13, 13)

    elif name == "squeezenet_split-37_0":
        mod, params = relay.testing.squeezenet_split.get_workload(N=37, npartition=0, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size, 1000, 13, 13)
    elif name == "squeezenet_split-37_1":
        mod, params = relay.testing.squeezenet_split.get_workload(N=37, npartition=1, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size, 1000, 13, 13)

    elif name == "squeezenet_split-38_0":
        mod, params = relay.testing.squeezenet_split.get_workload(N=38, npartition=0, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        output_shape = (batch_size, 1000, 1, 1)
    elif name == "squeezenet_split-38_1":
        mod, params = relay.testing.squeezenet_split.get_workload(N=38, npartition=1, batch_size=batch_size, dtype=dtype, image_shape=image_shape)
        input_shape = (batch_size, 1000, 1, 1)


    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)

    elif name == "inception_v3_split-5_0":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3_split.get_workload(N=5, npartition=0, batch_size=batch_size, dtype=dtype)
        output_shape = (batch_size, 80, 73,73)
    elif name == "inception_v3_split-5_1":
        mod, params = relay.testing.inception_v3_split.get_workload(N=5, npartition=1, batch_size=batch_size, dtype=dtype)
        input_shape  = (batch_size, 80, 73,73)

    elif name == "inception_v3_split-88_0":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3_split.get_workload(N=88, npartition=0, batch_size=batch_size, dtype=dtype)
        output_shape = (batch_size, 768, 17,17)
    elif name == "inception_v3_split-88_1":
        mod, params = relay.testing.inception_v3_split.get_workload(N=88, npartition=1, batch_size=batch_size, dtype=dtype)
        input_shape  = (batch_size, 768, 17,17)

    elif name == "inception_v3_split-107_0":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3_split.get_workload(N=107, npartition=0, batch_size=batch_size, dtype=dtype)
        output_shape = (batch_size, 2048,8,8)
    elif name == "inception_v3_split-107_1":
        mod, params = relay.testing.inception_v3_split.get_workload(N=107, npartition=1, batch_size=batch_size, dtype=dtype)
        input_shape  = (batch_size, 2048,8,8)


    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)

    elif name == "mlp":
        mod, params = relay.testing.mlp.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000
        )
    elif name == "vgg-19_split-20_0":
        mod, params = relay.testing.vgg_split.get_workload(N=20, npartition=0, batch_size=batch_size, dtype=dtype)
        output_shape = (batch_size, 512, 14, 14) 
    elif name == "vgg-19_split-20_1":
        mod, params = relay.testing.vgg_split.get_workload(N=20, npartition=1, batch_size=batch_size, dtype=dtype)
        input_shape = (batch_size, 512, 14, 14)

    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "densenet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.densenet.get_workload(
            densenet_size=n_layer, batch_size=batch_size, dtype=dtype
        )
    else:
        raise ValueError("Network not found.")

    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse

        mod, params = convert_model_dense_to_sparse(mod, params, random_params=True)

    return mod, params, input_shape, output_shape


def load_tflite_model(tflite_model_file, SIZE):
    # https://tvm.apache.org/docs/tutorials/frontend/from_tflite.html
    # https://tvm.apache.org/docs/tutorials/frontend/deploy_prequantized_tflite.html

    print("[INFO] Loading model from .tflite")
    tflite_model_buf = open(tflite_model_file, "rb").read()

    # Get TFLite model from buffer
    try:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    # ---------- Take input tensor name, shape and type, etc from TFlite functions...
    # TFLite input tensor name, shape and type
    #input_tensor = "input"
    input_shape = (1, SIZE, SIZE, 3)
    try:
        from tensorflow import lite as interpreter_wrapper
    except ImportError:
        from tensorflow.contrib import lite as interpreter_wrapper

    interpreter = interpreter_wrapper.Interpreter(model_content=tflite_model_buf)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    #print(input_details)
    output_shape = output_details[0]["shape"]
    input_tensor = input_details[0]["name"]
    input_dtype = input_details[0]["dtype"].__name__
    print(" \n ---- Using '{}' as input tensor, with '{}' data format ---- \n".format(input_tensor, input_dtype))
    ## --------------------------------------------

    # Parse TFLite model and convert it to a Relay module
    from tvm import relay, transform

    mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
    )
    return mod, params, input_tensor, input_dtype, output_shape
 
   
def load_onnx_model(onnx_model_file):
    print("[INFO] Loading model from .onnx")
    import onnx
    onnx_model = onnx.load(onnx_model_file)
    input_all = onnx_model.graph.input
    input_tensor = input_all[0].name
    SIZE = input_all[0].type.tensor_type.shape.dim[2].dim_value
    #input_tensor = "data"
    #types =   [ "float" , "int32" , "string" , "bool" , "uint8", "int8" , "uint16" ,"int16" , "int64" , "float16", "double"]
    input_dtype = "float32"; #print(" Warning data type manually set ")
    input_shape = (1, 3, SIZE, SIZE)
    shape_dict = {input_tensor: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return mod, params, input_tensor, input_shape, input_dtype 



## from https://tvm.apache.org/docs/tutorials/autotvm/tune_relay_x86.html
def evaluate_performance_N(module, dev, input_name="input", data_shape=(1, 224, 224, 3), dtype= "float32", N=0):
    # upload parameters to device
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module.set_input(input_name, data_tvm)

    # evaluate
    print("Evaluate inference time cost...")
    if N:    print(module.benchmark(dev, number=1, repeat=N))
    else:    print(module.benchmark(dev, number=5, min_repeat_ms=500))


def transform_image(image, dtype):
    image_data = np.asarray(image).astype(dtype)

    # Add a dimension to the image so that we have NHWC format layout
    image_data = np.expand_dims(image_data, axis=0)

    # Preprocess image as described here:
    # https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243
    image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
    image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
    image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
    return image_data


def load_real_image(img_path, im_height, im_width, dtype):
    from PIL import Image
    image = Image.open(img_path).resize((im_height, im_width))
    x = np.array(image).astype(dtype)
    data = np.reshape(x, (1, im_height, im_width, 3))
    return data



