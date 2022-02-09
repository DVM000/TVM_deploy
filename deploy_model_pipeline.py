
# https://tvm.apache.org/docs/tutorials/frontend/from_tflite.html
# https://tvm.apache.org/docs/tutorials/frontend/deploy_model_on_rasp.html

 
import sys
import os
import tvm
import numpy as np

from tvm import relay, transform
from tvm.contrib import graph_executor

from threading import Thread
import threading
import time

from util import bcolors

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
      '-m', '--model', required=True, help='Network name in relay. Options include mobilenet_split-N, inception_v3_split-N, squeezenet_split-N' +
                                           '\n where N is the number of layers in 1st partition.' +
                                           '\n Use <network>_split-help to check options. Example: -m mobilenet_split-help', )
parser.add_argument(
      '--batch', default=1, help='Batch size')
parser.add_argument(
       '--target', required=False, default="2,0", help='List of indexes for target devices hosting each partition. Use --target "help" to check enabled targets')
#parser.add_argument(
#      '--layout', default="NCHW", choices=("NCHW", "NHWC"), help='Layout for etworks from relay. {NCHW, NHWC}') # depends on implemented nework partitions
parser.add_argument(
      '-c', '--count', type=int, default=1000,
      help='Number of times to run inference')
parser.add_argument(
      '--printinfo', help='To test script on only one device', action="store_true", default=False)
parser.add_argument(
       '--logfile', required=False, default="_,_", help='Comma-separated list of Tuning log files. Use _,_ for not loading log_file')

args = parser.parse_args()
model_file = args.model; network = args.model
batch_size = int(args.batch)
MAX_N = args.count
#layout = args.layout 
log_file = [k for k in args.logfile.split(',')]

PRINT_INFO = args.printinfo  

####
import tvm.testing
target_list = tvm.testing.enabled_targets()

if args.target == "help":
    print(" -------------------------------------- \n INFO: Local Enabled targets: \n")
    import tvm.testing
    for i,t in enumerate(target_list):
        print('\t{})  {}'.format(i,t))
    print("  --------------------------------------")
    sys.exit(0)

target_idx = [int(k) for k in args.target.split(',')]

target_list_sel = []
for idx in target_idx:
    target_list_sel.append( target_list[idx] )


print(" -------------------------------------- ")
print(bcolors.WARNING + "\n Selected targets: \n" + bcolors.ENDC)
CONF = []
for t,d in target_list_sel:
    CONF.append( {'target': tvm.target.Target(t), 'dev': d})
    print(bcolors.WARNING + "({}) Target: {} \t Device: {}".format(len(CONF)-1,CONF[-1]['target'],CONF[-1]['dev']) + bcolors.ENDC)
print("  --------------------------------------")

if 'help' in model_file: 
    netname = model_file.split('-help')[0]
    model_file = netname + '-0'

from util import get_network
layout = "NCHW" 
dtype = "float32"
input_tensor = "data"
#mod, params, input_shape, output_shape = get_network(model_file, batch_size=1, dtype=dtype, layout=layout)
mod1, params1, input_shape1, output_shape1 = get_network(model_file+'_0', batch_size=batch_size, dtype=dtype, layout=layout)
mod2, params2, input_shape2, output_shape2 = get_network(model_file+'_1', batch_size=batch_size, dtype=dtype, layout=layout)

mod = [mod1, mod2]
params = [params1, params2]
input_shape = [input_shape1, input_shape2]
output_shape = [output_shape1, output_shape2]

data_in = tvm.nd.array((np.random.uniform(size=input_shape[0])).astype(dtype))
data_in2 = tvm.nd.array((np.random.uniform(size=input_shape[1])).astype(dtype))


## Build libraries and create graph_executor
## ----------------------------------------------------------------------------------------
input_data_filled = []
output_consumed = []
print("\n -- Building graphs -- \n")
for i,c in enumerate(CONF):
    print("  Using Target {} and device {}".format(c['target'], c['dev']))
    if log_file[i] == "_":
        with transform.PassContext(opt_level=3):
            lib = relay.build(mod[i], c['target'], params=params[i])
    else:
        from tvm import auto_scheduler
        print("  Compiling with tuning log_file {} ".format(log_file[i]))
        with auto_scheduler.ApplyHistoryBest(log_file[i]):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod[i], c['target'], params=params[i])

    c['module'] = graph_executor.GraphModule(lib["default"](c['dev'])) 

    input_data_filled.append( threading.Event() )
    input_data_filled[i].clear() # waits for new input data

    output_consumed.append( threading.Event() )
    output_consumed[i].set() # input data has been taken

        
class subGraph:
    """
    Class to execute model partition (#idx) in a new Python Thread.
    - Class that successively run TVM workload, when a new input is available (input_data_filled[idx] is set). 
    - Once read the input, it is again ready for receiving new inputs (input_data_filled[idx] is clean, output_consumed[idx] is set)
    - Workload in the loop is only executed when previous iteration's output data have been externally fetched (output_consumed[idx+1] is set)

    Parameters
    ----------
    module : GraphModule
        TVM module definig the graph.
    dev: target device.
    input_tensor : string
        name of the graph input node
    idx : int
        Thread event identifier, to read and transfer data
    key : string
        key name to be assigned to this subgraph
    autoinput : boolean
        True for the 1st subgraph of the workload partition
    """

    def __init__(self, module, dev, input_tensor, idx, key = 0, autoinput = 0):
        self.key = key
        self.module = module
        self.module.run() ## running 1st inference
        self.dev = dev
        self.input_tensor = input_tensor
        self.input_data = None 
        self.idx = idx
        self.output = None 
        self.status = 0 # 0: waiting, 1: running, -1: stopped
        self.N = 0
        self.mean_t = []
        self.color = bcolors.OKBLUE if idx==0 else bcolors.OKGREEN
        if PRINT_INFO:  print(self.color + "[{}] >> created subGraph".format(self.key)+ bcolors.ENDC)

        self.autoinput = autoinput # 1: for 1st device, 0: otherwise
        self.subGraph2 = 0 # subGraph object concening the following device 

    def set_2sub(self, subGraph2):
        self.subGraph2 = subGraph2

    def start(self):
        self.th = Thread(target=self.run_, args=())
        self.th.start()
 

    def run_(self):

        global t0, tt 
        self.status = 1
        while( self.N < MAX_N and self.status != -1):
            # -- a) waits for new input data -- #            
            input_data_filled[self.idx].wait()
            output_consumed[self.idx].clear() # received but still need to read data and set it as module input

            # b) -- Take and process data -- #
            self.trun0 = time.time()
            if PRINT_INFO:  print(self.color + "img#{} [ {:.1f}           {}".format(self.N, 1e3*(self.trun0-t0), self.input_data.shape) + bcolors.ENDC)

            self.module.set_input(self.input_tensor, self.input_data) #tvm.nd.array(self.input_data))

            # c) -- Let receive new input again -- #
            output_consumed[self.idx].set()
            if not self.autoinput: # if not autoinput
                input_data_filled[self.idx].clear()    

            # b) -- ... and process data -- #     
            self.module.run()
            self.output = self.module.get_output(0).numpy() 
            #print(self.output.shape)
            self.trun1 = time.time()
            self.mean_t.append(self.trun1-self.trun0)
            if PRINT_INFO:  print(self.color + "              {:.1f} ]  ({:.1f} ms/run)".format(1e3*(self.trun1-t0), 1e3*(self.trun1-self.trun0)) + bcolors.ENDC)

            # d) -- Waits for data being fecthed (not using any queue)-- #
            if self.subGraph2 !=0: # if there is a second graph
                #print(self.color + "[{}] >> waiting before sending img {}".format(self.key, self.N) + bcolors.ENDC); 
                output_consumed[self.subGraph2.idx].wait()
                self.subGraph2.set_input( tvm.nd.array(self.output) )
            else:
                tt.append( 1e3*(time.time() -t0) )
  
            if PRINT_INFO:  print(self.color + "              {:.1f} output #{} delivered ".format(1e3*(time.time()-t0), self.N) + bcolors.ENDC)

            self.N += 1

        print(self.color + "[{}] >> {} executions -- avg {:.2f} ms/run\n".format(self.key, self.N, 1e3*np.mean(self.mean_t)) + bcolors.ENDC)

        self.status = -1 
        if self.subGraph2 ==0:  print(bcolors.HEADER + "Average throughput {:.1f} fps ({:.2f} ms/img) \t-- elapsed time {:.2f} s \n".format( 1e3/np.mean(np.diff(tt)), np.mean(np.diff(tt)), time.time()-t0) + bcolors.ENDC)


    def get_status(self):
        return self.status

    def stop(self):
        self.status = -1
        print(self.color + "[{}] >> {} executions \n".format(self.key, self.N) + bcolors.ENDC)

    def set_input(self, input_data):
        self.input_data = input_data
        input_data_filled[self.idx].set()
        output_consumed[self.idx].clear()

    def get_output(self):
        return self.output

    def set_output(self, outdata):
        self.output = outdata


tt = [] # output time instants

runner = []

for i,c in enumerate(CONF):
    runner.append( subGraph(c['module'], c['dev'], input_tensor, idx=i, key='dev-{}'.format(i), autoinput=1-i) )


runner[0].set_2sub(runner[1])
for r in runner:
    r.start() # waiting data

print("\n Starting -------------------------------------- \n ")
t0 = time.time()
runner[0].set_input(data_in)






