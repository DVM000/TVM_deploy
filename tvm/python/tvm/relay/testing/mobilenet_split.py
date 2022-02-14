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
"""
Port of NNVM version of MobileNet to Relay.
"""
# pylint: disable=invalid-name

from tvm import relay
from . import layers
from .init import create_workload


def conv_block(
    data,
    name,
    channels,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding=(1, 1),
    epsilon=1e-5,
    layout="NCHW",
):
    """Helper function to construct conv_bn-relu"""
    # convolution + bn + relu
    conv = layers.conv2d(
        data=data,
        channels=channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
        name=name + "_conv",
    )
    bn = layers.batch_norm_infer(data=conv, epsilon=epsilon, name=name + "_bn")
    act = relay.nn.relu(data=bn)
    return act


def separable_conv_block(
    data,
    name,
    depthwise_channels,
    pointwise_channels,
    kernel_size=(3, 3),
    downsample=False,
    padding=(1, 1),
    epsilon=1e-5,
    layout="NCHW",
    dtype="float32",
):
    """Helper function to get a separable conv block"""
    if downsample:
        strides = (2, 2)
    else:
        strides = (1, 1)

    # depthwise convolution + bn + relu
    if layout == "NCHW":
        wshape = (depthwise_channels, 1) + kernel_size
    elif layout == "NHWC":
        wshape = kernel_size + (depthwise_channels, 1)
    else:
        raise ValueError("Invalid layout: " + layout)
    bn_axis = layout.index("C")
    weight = relay.var(name + "_weight", shape=wshape, dtype=dtype)
    conv1 = layers.conv2d(
        data=data,
        weight=weight,
        channels=depthwise_channels,
        groups=depthwise_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout, True),
        name=name + "_depthwise_conv1",
    )
    bn1 = layers.batch_norm_infer(data=conv1, epsilon=epsilon, axis=bn_axis, name=name + "_bn1")
    act1 = relay.nn.relu(data=bn1)
    # pointwise convolution + bn + relu
    conv2 = layers.conv2d(
        data=act1,
        channels=pointwise_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding=(0, 0),
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
        name=name + "_conv2",
    )
    bn2 = layers.batch_norm_infer(data=conv2, epsilon=epsilon, axis=bn_axis, name=name + "_bn2")
    act2 = relay.nn.relu(data=bn2)
    return act2

def separable_conv_block_A(
    data,
    name,
    depthwise_channels,
    pointwise_channels,
    kernel_size=(3, 3),
    downsample=False,
    padding=(1, 1),
    epsilon=1e-5,
    layout="NCHW",
    dtype="float32",
):
    """Helper function to get a separable conv block (part A)"""
    if downsample:
        strides = (2, 2)
    else:
        strides = (1, 1)

    # depthwise convolution + bn + relu
    if layout == "NCHW":
        wshape = (depthwise_channels, 1) + kernel_size
    elif layout == "NHWC":
        wshape = kernel_size + (depthwise_channels, 1)
    else:
        raise ValueError("Invalid layout: " + layout)
    bn_axis = layout.index("C")
    weight = relay.var(name + "_weight", shape=wshape, dtype=dtype)
    conv1 = layers.conv2d(
        data=data,
        weight=weight,
        channels=depthwise_channels,
        groups=depthwise_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout, True),
        name=name + "_depthwise_conv1",
    )
    bn1 = layers.batch_norm_infer(data=conv1, epsilon=epsilon, axis=bn_axis, name=name + "_bn1")
    act1 = relay.nn.relu(data=bn1)
 
    return act1

def separable_conv_block_B(
    data,
    name,
    depthwise_channels,
    pointwise_channels,
    kernel_size=(3, 3),
    downsample=False,
    padding=(1, 1),
    epsilon=1e-5,
    layout="NCHW",
    dtype="float32",
):
    """Helper function to get a separable conv block (part B)"""
    if downsample:
        strides = (2, 2)
    else:
        strides = (1, 1)

    if layout == "NCHW":
        wshape = (depthwise_channels, 1) + kernel_size
    elif layout == "NHWC":
        wshape = kernel_size + (depthwise_channels, 1)
    else:
        raise ValueError("Invalid layout: " + layout)
    bn_axis = layout.index("C")
    weight = relay.var(name + "_weight", shape=wshape, dtype=dtype)

    # pointwise convolution + bn + relu
    conv2 = layers.conv2d(
        data=data,
        channels=pointwise_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding=(0, 0),
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
        name=name + "_conv2",
    )
    bn2 = layers.batch_norm_infer(data=conv2, epsilon=epsilon, axis=bn_axis, name=name + "_bn2")
    act2 = relay.nn.relu(data=bn2)
    return act2



def mobile_net(
    N=23,
    num_classes=1000,
    data_shape=(1, 3, 224, 224),
    dtype="float32",
    alpha=1.0,
    is_shallow=False,
    layout="NCHW",
):
    """Function to construct a MobileNet"""
    data = relay.var("data", shape=data_shape, dtype=dtype)
    body = conv_block(data, "conv_block_1", int(32 * alpha), strides=(2, 2), layout=layout)
    body = separable_conv_block(
        body, "separable_conv_block_1", int(32 * alpha), int(64 * alpha), layout=layout, dtype=dtype
    )
    if N == 4:
        body = separable_conv_block_A(
            body,
            "separable_conv_block_2_a",
            int(64 * alpha),
            int(128 * alpha),
            downsample=True,
            layout=layout,
            dtype=dtype,
        )
        body1 = body
        data2 = relay.var("data", shape=(data_shape[0],64,56,56), dtype=dtype)
        body = separable_conv_block_B(
            data2,
            "separable_conv_block_2_b",
            int(64 * alpha),
            int(128 * alpha),
            downsample=True,
            layout=layout,
            dtype=dtype,
        )
    else:
        body = separable_conv_block(
            body,
            "separable_conv_block_2",
            int(64 * alpha),
            int(128 * alpha),
            downsample=True,
            layout=layout,
            dtype=dtype,
        )
    if N==5: 
        body1 = body
        body = relay.var("data", shape=(data_shape[0],128,56,56), dtype=dtype)
    
    if N==6:
        body = separable_conv_block_A(
            body,
            "separable_conv_block_3_a",
            int(128 * alpha),
            int(128 * alpha),
            layout=layout,
            dtype=dtype,
        )
        body1 = body
        data2 = relay.var("data", shape=(data_shape[0],128,56,56), dtype=dtype)
        body = separable_conv_block_B(
            data2,
            "separable_conv_block_3_b",
            int(128 * alpha),
            int(128 * alpha),
            layout=layout,
            dtype=dtype,
        )
    else:
        body = separable_conv_block(
            body,
            "separable_conv_block_3",
            int(128 * alpha),
            int(128 * alpha),
            layout=layout,
            dtype=dtype,
        )
    body = separable_conv_block(
        body,
        "separable_conv_block_4",
        int(128 * alpha),
        int(256 * alpha),
        downsample=True,
        layout=layout,
        dtype=dtype,
    )
    body = separable_conv_block(
        body,
        "separable_conv_block_5",
        int(256 * alpha),
        int(256 * alpha),
        layout=layout,
        dtype=dtype,
    )
    body = separable_conv_block(
        body,
        "separable_conv_block_6",
        int(256 * alpha),
        int(512 * alpha),
        downsample=True,
        layout=layout,
        dtype=dtype,
    )
    if is_shallow:
        body = separable_conv_block(
            body,
            "separable_conv_block_7",
            int(512 * alpha),
            int(1024 * alpha),
            downsample=True,
            layout=layout,
            dtype=dtype,
        )
        body = separable_conv_block(
            body,
            "separable_conv_block_8",
            int(1024 * alpha),
            int(1024 * alpha),
            downsample=True,
            layout=layout,
            dtype=dtype,
        )
    else:
        for i in range(7, 12):
            body = separable_conv_block(
                body,
                "separable_conv_block_%d" % i,
                int(512 * alpha),
                int(512 * alpha),
                layout=layout,
                dtype=dtype,
            )
        if N==23: 
            body1 = body
            body = relay.var("data", shape=(data_shape[0],512,14,14), dtype=dtype)

        body = separable_conv_block(
            body,
            "separable_conv_block_12",
            int(512 * alpha),
            int(1024 * alpha),
            downsample=True,
            layout=layout,
            dtype=dtype,
        )
        if N==25: 
            body1 = body
            body = relay.var("data", shape=(data_shape[0],1024,7,7), dtype=dtype)

        body = separable_conv_block(
            body,
            "separable_conv_block_13",
            int(1024 * alpha),
            int(1024 * alpha),
            layout=layout,
            dtype=dtype,
        )
    if N==27: 
        body1 = body
        body= relay.var("data", shape=(data_shape[0],1024,7,7), dtype=dtype)

    pool = relay.nn.global_avg_pool2d(data=body, layout=layout)
    flatten = relay.nn.batch_flatten(data=pool)
    weight = relay.var("fc_weight")
    bias = relay.var("fc_bias")
    #data2 = relay.var("data", shape=(data_shape[0],1024), dtype=dtype)
    fc = relay.nn.dense(data=flatten, weight=weight, units=num_classes)
    fc = relay.nn.bias_add(fc, bias)
    if N==30: 
        body1 = fc
        fc = relay.var("data", shape=(data_shape[0],1000), dtype=dtype)
    softmax = relay.nn.softmax(data=fc)
       
    return [relay.Function(relay.analysis.free_vars(body1), body1), relay.Function(relay.analysis.free_vars(softmax), softmax)]



def get_workload(
    N=23, npartition=0, batch_size=1, num_classes=1000, image_shape=(3, 224, 224), dtype="float32", layout="NCHW"
):
    """Get benchmark workload for mobilenet

    Parameters
    ----------
    batch_size : int, optional
        The batch size used in the model

    num_classes : int, optional
        Number of classes

    image_shape : tuple, optional
        The input image shape, cooperate with layout

    dtype : str, optional
        The data type

    layout : str, optional
        The data layout of image_shape and the operators
        cooperate with image_shape

    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains a MobileNet network.

    params : dict of str to NDArray
        The parameters.
    """
    if N != 4 and N != 5 and N != 6 and N != 23 and N != 25 and N != 27 and N != 30:
        print('mobilenet partition with {} layers not implemented'.format(N))
        import sys; sys.exit(1)
    data_shape = tuple([batch_size] + list(image_shape))
    net = mobile_net(
        N=N,
        num_classes=num_classes,
        data_shape=data_shape,
        dtype=dtype,
        alpha=1.0,
        is_shallow=False,
        layout=layout,
    )
   
    #return [create_workload(n) for n in net]
    return create_workload(net[npartition])
