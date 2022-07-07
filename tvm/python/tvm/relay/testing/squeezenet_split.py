# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=unused-argument

"""
Symbol of SqueezeNet

Reference:
Iandola, Forrest N., et al.
"Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size." (2016).
"""

from tvm import relay
from .init import create_workload
from . import layers

# Helpers
def _make_fire(net, squeeze_channels, expand1x1_channels, expand3x3_channels, prefix):
    net = _make_fire_conv(net, squeeze_channels, 1, 0, "%s_input" % prefix)

    left = _make_fire_conv(net, expand1x1_channels, 1, 0, "%s_left" % prefix)
    right = _make_fire_conv(net, expand3x3_channels, 3, 1, "%s_right" % prefix)
    # NOTE : Assume NCHW layout here
    net = relay.concatenate((left, right), axis=1)
    return net


def _make_fire_conv(net, channels, kernel_size, padding=0, prefix=""):
    net = layers.conv2d(
        net,
        channels=channels,
        kernel_size=(kernel_size, kernel_size),
        padding=(padding, padding),
        name="%s_conv" % prefix,
    )
    net = relay.nn.bias_add(net, relay.var("%s_conv_bias" % prefix))
    net = relay.nn.relu(net)
    return net


# Net
def get_net(N, batch_size, image_shape, num_classes, version, dtype):
    """Get symbol of SqueezeNet

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    image_shape : tuple, optional
        The input image shape

    num_classes: int
        The number of classification results

    version : str, optional
        "1.0" or "1.1" of SqueezeNet
    """
    assert version in [
        "1.0",
        "1.1",
    ], "Unsupported SqueezeNet version {version}:" "1.0 or 1.1 expected".format(version=version)
    data_shape = (batch_size,) + image_shape
    net = relay.var("data", shape=data_shape, dtype=dtype)
    '''if version == "1.0":
        net = layers.conv2d(
            net, channels=96, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3), name="conv1"
        )
        net = relay.nn.bias_add(net, relay.var("conv1_bias"))
        net = relay.nn.relu(net)
        net = relay.nn.max_pool2d(net, pool_size=(3, 3), strides=(2, 2))
        net = _make_fire(net, 16, 64, 64, "fire1")
        net = _make_fire(net, 16, 64, 64, "fire2")
        net = _make_fire(net, 32, 128, 128, "fire3")
        net = relay.nn.max_pool2d(net, pool_size=(3, 3), strides=(2, 2))
        net = _make_fire(net, 32, 128, 128, "fire4")
        net = _make_fire(net, 48, 192, 192, "fire5")
        net = _make_fire(net, 48, 192, 192, "fire6")
        net = _make_fire(net, 64, 256, 256, "fire7")
        net = relay.nn.max_pool2d(net, pool_size=(3, 3), strides=(2, 2))
        net = _make_fire(net, 64, 256, 256, "fire8")
    else:'''
    net = layers.conv2d(
            net, channels=64, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), name="conv1"
        )
    net = relay.nn.bias_add(net, relay.var("conv1_bias"))
    net = relay.nn.relu(net)
    net = relay.nn.max_pool2d(net, pool_size=(3, 3), strides=(2, 2))
    if N == 2: 
        net1 = net
        net = relay.var("data", shape=(batch_size, 64, 55, 55), dtype=dtype)

    net = _make_fire(net, 16, 64, 64, "fire1")
    if N == 6: 
        net1 = net
        net = relay.var("data", shape=(batch_size, 128, 55, 55), dtype=dtype)
        
    net = _make_fire(net, 16, 64, 64, "fire2")
    net = relay.nn.max_pool2d(net, pool_size=(3, 3), strides=(2, 2))
    if N == 10: 
        net1 = net
        net = relay.var("data", shape=(batch_size, 128, 27, 27), dtype=dtype)

    net = _make_fire(net, 32, 128, 128, "fire3")
    net = _make_fire(net, 32, 128, 128, "fire4")
    #net = relay.var("data", shape=(batch_size, 256, 27, 27), dtype=dtype)
    net = relay.nn.max_pool2d(net, pool_size=(3, 3), strides=(2, 2))
    net = _make_fire(net, 48, 192, 192, "fire5")
    if N == 24: 
        net1 = net
        net = relay.var("data", shape=(batch_size, 384, 13, 13), dtype=dtype)

    net = _make_fire(net, 48, 192, 192, "fire6")
    net = _make_fire(net, 64, 256, 256, "fire7")
    if N == 32: 
        net1 = net
        net = relay.var("data", shape=(batch_size, 512, 13, 13), dtype=dtype)
        
    net = _make_fire(net, 64, 256, 256, "fire8")
    net = relay.nn.dropout(net, rate=0.5)
    #net = relay.var("data", shape=(batch_size, 512, 13, 13), dtype=dtype)
    net = layers.conv2d(net, channels=num_classes, kernel_size=(1, 1), name="conv_final")
    if N == 36:
        net1 = net
        net = net = relay.var("data", shape=(batch_size, 1000, 13, 13), dtype=dtype)

    net = relay.nn.bias_add(net, relay.var("conv_final_bias"))
    net = relay.nn.relu(net)
    if N == 37:
        net1 = net
        net = net = relay.var("data", shape=(batch_size, 1000, 13, 13), dtype=dtype)

    net = relay.nn.global_avg_pool2d(net)
    if N == 38:
        net1 = net
        net = net = relay.var("data", shape=(batch_size, 1000, 1, 1), dtype=dtype)

    net = relay.nn.batch_flatten(net)
    net = relay.nn.softmax(net)

    return [relay.Function(relay.analysis.free_vars(net1), net1), relay.Function(relay.analysis.free_vars(net), net)]


def get_workload(
    N=32, npartition=0, batch_size=1, num_classes=1000, version="1.0", image_shape=(3, 224, 224), dtype="float32"
):
    """Get benchmark workload for SqueezeNet

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of classes

    version : str, optional
        "1.0" or "1.1" of SqueezeNet

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains a SqueezeNet network.

    params : dict of str to NDArray
        The parameters.
    """
    if N not in [2, 6, 10, 24, 32, 36, 37, 38]:
        print('squeezenet partition with {} layers not implemented'.format(N))
        import sys; sys.exit(1)
    net = get_net(N, batch_size, image_shape, num_classes, version, dtype)
    return create_workload(net[npartition]) #create_workload(net)
