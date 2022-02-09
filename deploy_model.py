# https://tvm.apache.org/docs/tutorials/frontend/from_tflite.html
# https://tvm.apache.org/docs/tutorials/frontend/deploy_model_on_rasp.html
  
import sys
import os
import tvm
import numpy as np
from util import bcolors

num_threads = 4
os.environ["TVM_NUM_THREADS"] = str(num_threads)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
      '-r', '--remote', help='Whether to run models on a remote ARM device or not. \n Specify remote device either with IP, port; or with key', action="store_true", default=False)
parser.add_argument(
      '--profile', help='Whether to profile the model', action="store_true", default=False)
parser.add_argument(
      '-m', '--model', required=True, help='Either network name in relay or .tflite/.onnx file')
parser.add_argument(
      '--size', default=224, help='Input height and width')
parser.add_argument(
      '--batch', default=1, help='Batch size')
parser.add_argument(
       '--target', required=False, default="llvm", help='Target {"llvm", "cuda", "llvm -mtriple=aarch64-linux-gnu", etc}. Use --target "help" to check enabled targets')
parser.add_argument(
      '--layout', default="NCHW", choices=("NCHW", "NHWC"), help='Layout for etworks from relay. {NCHW, NHWC}')
parser.add_argument(
       '--key', required=False, default="", help='Remote RPC-key. If not specified, remote connection will be enabled using port and host. (Assuming tracker on 127.0.0.1:9190)')
parser.add_argument(
       '--host', required=False, default="192.168.41.53", help='Remote IP')
parser.add_argument(
       '--port', required=False, default="9090", help='Remote RPC-Server Port')
parser.add_argument(
      '-c', '--count', type=int, default=0,
      help='Number of times to run inference. If zero, benchmark will take at least min_repeat_ms=500')
parser.add_argument(
       '--logfile', required=False, default="", help='Tuning log file')

args = parser.parse_args()
model_file = args.model; network = args.model
Nexe = args.count
target_ = args.target
SIZE = int(args.size)
batch_size = int(args.batch)
layout = args.layout 
host = args.host 
port = int(args.port) 
key = args.key
local_demo = not args.remote
log_file = args.logfile
PROFILE = args.profile

if not local_demo:
    if key=="":    
        print("\n ** ---- Remote execution on {}:{} ---- ** \n".format(host, port))
        print(bcolors.WARNING + "\n\t [INFO] Please execute RPC server on device: python -m tvm.exec.rpc_server --host 0.0.0.0 --port={}  \n".format(port) + bcolors.ENDC)   
    else:           
        print("\n ** ---- Remote execution on device with rpc-key='{}' ---- ** \n".format(key))
        print(bcolors.WARNING + "\n\t [INFO] Please execute RPC server on device: python -m tvm.exec.rpc_server --tracker=IP:PORT --key={}  \n".format(key) + bcolors.ENDC)


if target_ == "help":
    print(" -------------------------------------- \n INFO: Local Enabled targets: \n")
    import tvm.testing
    for i,t in enumerate(tvm.testing.enabled_targets()):
        print('\t{})  {}'.format(i,t))
    print("  --------------------------------------")
    sys.exit(0)


# --- LOAD MODEL ---
# -------------------------------------------------------------
if "inception" in network:  SIZE = 299

print(' --- Loading model {} ---'.format(model_file))
if ".tflite" in model_file:
    from util import load_tflite_model
    mod, params, input_tensor, dtype, output_shape = load_tflite_model(model_file, SIZE)

elif ".onnx" in model_file:
    from util import load_onnx_model
    mod, params, input_tensor, input_shape, dtype = load_onnx_model(model_file)
else:
    from util import get_network
    dtype = "float32"
    input_tensor = "data"
    mod, params, input_shape, output_shape = get_network(model_file, batch_size=batch_size, dtype=dtype, layout=layout)


## Change layout: 
#https://tvm.apache.org/docs/dev/convert_layout.html
if (".tflite" in model_file or ".onnx" in model_file) and layout=='NCHW':
    print("[INFO] --- Changing CNN {} layout to {} layout... ---".format(network, layout))
    from tvm import relay
    # We assume our model's heavily-layout sensitive operators only consist of nn.conv2d
    desired_layouts = {'nn.conv2d': [layout, 'default'],
                      'qnn.conv2d': [layout, 'default']
                      }
    # Convert the layout to NCHW
    # RemoveUnunsedFunctions is used to clean up the graph.
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)


# --- LOAD INPUT DATA ---
# -------------------------------------------------------------
print(' --- Loading test image ---') 
# set random input 
image_data = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
print('Using random data as input')
print("input shape: ", image_data.shape)


# --- BUILD THE MODEL FOR A SPECIFIC TARGET ---
# -------------------------------------------------------------
target = tvm.target.Target(target_)
print(" --- Target {} --> {} ---".format(target_, target))

from tvm import relay, transform
if log_file == "":
    with transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params) 
else:
    from tvm import auto_scheduler
    print("Compile with tuning log_file {}...".format(log_file))
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)

# Save the library at local temporary directory.
from tvm.contrib import utils
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.tar")
lib.export_library(lib_fname)


# --- RUN THE MODEL ---
# -------------------------------------------------------------
print(' --- Executing the graph... ---')
from tvm import te
from tvm.contrib import graph_executor as runtime
from tvm import rpc


if local_demo:
    # Create a runtime executor module
    if "cuda" in target_:
        dev = tvm.device(str(target_), 0)
    else:
        dev = tvm.cpu()
    module = runtime.GraphModule(lib["default"](dev))   

else: # obtain an RPC session from remote device.
    print(' --- Establishing connection... ---')
    if key=="":
        remote = rpc.connect(host, port)
        print(" -- sucessful connection to device {}:{}".format(host,port))
    else: 
        # https://discuss.tvm.apache.org/t/android-rpc-requesting-remote-session-from-rpc-tracker-failed/8413
        # Establish remote connection with target hardware
        tracker = rpc.connect_tracker("127.0.0.1", 9190)
        remote = tracker.request(key, priority=0, session_timeout=60)
        print(" -- sucessful connection to device with rpc-key={}".format(key))

    # upload the library to remote device and load it
    remote.upload(lib_fname)
    rlib = remote.load_module("net.tar")

    # create the remote runtime module
    if "opencl" in target_:  dev = remote.cl()
    elif "cuda" in target_:
        dev = remote.device(str(target_), 0)
    else:
        dev = remote.cpu(0)
    module = runtime.GraphModule(rlib["default"](dev))


# --- MEASURE INFERENCE TIME ---
# -------------------------------------------------------------
import time

print("\n Starting -------------------------------------- \n ")

if not PROFILE:
    # a) module.benchmark()
    #from util import evaluate_performance_N
    #evaluate_performance_N(module, dev, input_name=input_tensor, data_shape=input_shape, dtype= dtype, N=Nexe)

    # b) Run Nexe times
    module.set_input(input_tensor, tvm.nd.array(image_data))
    module.run()
    tvm_output = module.get_output(0).numpy()

    '''t0 = time.time()
    for k in range(Nexe):
        module.run()
    print("Average on {} executions: {:.2f} ms  -- just run -- \n".format(Nexe, 1e3*(time.time()-t0)/Nexe))
    print("----------------------------------------------------------------------------------------\n ")'''

    # c) Feed input data +  Run + Get output
    time.sleep(1)
    t0 = time.time()
    for k in range(Nexe):
        module.set_input(input_tensor, tvm.nd.array(image_data))
        module.run()
        tvm_output = module.get_output(0).numpy()
     
    print("Average on {} executions: {:.2f} ms  -- including set_input, run and get_output --\n".format(Nexe, 1e3*(time.time()-t0)/Nexe))
    print("----------------------------------------------------------------------------------------\n ")

    print('Output shape: {}'.format(tvm_output.shape))


# --- PROFILE PER-LAYER INFERENCE TIME ---
# -------------------------------------------------------------
#https://tvm.apache.org/docs/dev/debugger.html
if PROFILE:
    if not local_demo:
        print('[INFO] remote profiling not implemented'); sys.exit(0)

    from tvm.contrib.debugger import debug_executor
    json_ = lib.get_graph_json()

    lib.export_library("mod.so")
    lib2 = tvm.runtime.load_module("mod.so")
    m = debug_executor.create(json_, lib2, dev, dump_root="/tmp/tvmdbg", number=Nexe, repeat=1, min_repeat_ms=1) #lib["get_graph_json"](), lib, dev, dump_root="/tmp/tvmdbg")
    m.set_input(input_tensor, tvm.nd.array(image_data))
    m.set_input(**params)
    # execute
    m.run()
    tvm_output = m.get_output(0, tvm.nd.empty(output_shape, dtype)).numpy()


# --- GET OUTPUT ---
# -------------------------------------------------------------
predictions = np.squeeze(tvm_output)
print("TVM Top-5 labels:", predictions.argsort()[-5:][::-1] )

# Get top 1 prediction
prediction = np.argmax(predictions)

# Convert id to class name and show the result
print("The image prediction result is: id " + str(prediction))


sys.exit(0)



