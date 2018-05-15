# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>
# Reference code: https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope
import numpy as np

@add_arg_scope
def conv(x, filter_size, out_dim, 
         name='conv', stride=1, 
         padding='SAME',
         nl=tf.identity,
         data_dict=None,
         init_w=None, init_b=None,
         use_bias=True,
         wd=None,
         trainable=True):
    """ 
    2D convolution 

    Args:
        x (tf.tensor): a 4D tensor
           Input number of channels has to be known
        filter_size (int or list with length 2): size of filter
        out_dim (int): number of output channels
        name (str): name scope of the layer
        stride (int or list): stride of filter
        padding (str): 'VALID' or 'SAME' 
        init_w, init_b: initializer for weight and bias variables. 
           Default to 'random_normal_initializer'
        nl: a function

    Returns:
        tf.tensor with name 'output'
    """

    in_dim = int(x.shape[-1])
    assert in_dim is not None,\
    'Number of input channel cannot be None!'

    filter_shape = get_shape2D(filter_size) + [in_dim, out_dim]
    strid_shape = get_shape4D(stride)

    padding = padding.upper()

    convolve = lambda i, k: tf.nn.conv2d(i, k, strid_shape, padding)

    with tf.variable_scope(name) as scope:
        weights = new_weights('weights', 0, filter_shape, initializer=init_w,
                              data_dict=data_dict, trainable=trainable, wd=wd)
        out = convolve(x, weights)

        if use_bias:
            biases = new_biases('biases', 1, [out_dim], initializer=init_b,
                            data_dict=data_dict, trainable=trainable)
            out = tf.nn.bias_add(out, biases)
            # bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        
        output = nl(out, name = 'output')
        # output = nl(out)
        return output

@add_arg_scope
def dconv(x, filter_size, out_dim=None, 
         out_shape=None,
         out_shape_by_tensor=None,
         name='dconv', stride=2, 
         padding='SAME',
         nl=tf.identity,
         data_dict=None,
         init_w=None, init_b=None,
         wd=None,
         trainable=True):
    """ 
    2D deconvolution 

    Args:
        x (tf.tensor): a 4D tensor
           Input number of channels has to be known
        filter_size (int or list with length 2): size of filter
        out_dim (int): number of output channels
        out_shape (list(int)): shape of output without None
        out_shape_by_tensor (tf.tensor): a tensor has the same shape
                                         of output except the out_dim
        name (str): name scope of the layer
        stride (int or list): stride of filter
        padding (str): 'VALID' or 'SAME' 
        init: initializer for variables. Default to 'random_normal_initializer'
        nl: a function

    Returns:
        tf.tensor with name 'output'
    """
    stride = get_shape4D(stride)
    
    assert out_dim is not None or out_shape is not None\
    or out_shape_by_tensor is not None,\
    'At least one of (out_dim, out_shape_by_tensor, out_shape) \
    should be not None!'

    assert out_shape is None or out_shape_by_tensor is None,\
    'out_shape and out_shape_by_tensor cannot be both given!'

    in_dim = x.get_shape().as_list()[-1]

    # TODO other ways to determine the output shape 
    if out_shape_by_tensor is not None:
        if out_dim is None:
            out_dim = out_shape_by_tensor.get_shape().as_list()[-1]
        out_shape = tf.shape(out_shape_by_tensor)
        out_shape = tf.stack([out_shape[0], out_shape[1], 
                              out_shape[2], out_dim])
    elif out_shape is not None:
        if out_dim is None:
            out_dim = out_shape[-1]
        out_shape = tf.stack([out_shape[0], out_shape[1], 
                              out_shape[2], out_dim])
    else:
        x_shape = tf.shape(x)
        # assume output shape is input_shape*stride
        out_shape = tf.stack([x_shape[0], tf.multiply(x_shape[1], stride[1]), 
                        tf.multiply(x_shape[2], stride[2]), out_dim])

    filter_shape = get_shape2D(filter_size) + [out_dim, in_dim]

    with tf.variable_scope(name) as scope:

        weights = new_weights('weights', 0, filter_shape, initializer=init_w,
                             data_dict=data_dict, trainable=trainable, wd=wd)
        biases = new_biases('biases', 1, [out_dim], initializer=init_b,
                           data_dict=data_dict, trainable=trainable)
        dconv = tf.nn.conv2d_transpose(x, weights, 
                               output_shape=out_shape, 
                               strides=stride, 
                               padding=padding, 
                               name=scope.name)
        bias = tf.nn.bias_add(dconv, biases)
        # TODO need test
        bias.set_shape([None, None, None, out_dim])
        # if in_shape[1]:
        #     in_shape[1] *= stride[1]
        # if in_shape[2]:
        #     in_shape[2]*= stride[2]
        # bias.set_shape(in_shape)
        output = nl(bias, name='output')
        return output

@add_arg_scope
def fc(x, out_dim, name='fc', nl=tf.identity, 
       init_w=None, init_b=None,
       data_dict=None,
       wd=None, 
       trainable=True,
       re_dict=False):
    """ 
    Fully connected layer 

    Args:
        x (tf.tensor): a tensor to be flattened 
           The first dimension is the batch dimension
        num_out (int): dimension of output
        name (str): name scope of the layer
        init: initializer for variables. Default to 'random_normal_initializer'
        nl: a function

    Returns:
        tf.tensor with name 'output'
    """

    x_flatten = batch_flatten(x)
    # x_flatten = tf.reshape(x_flatten, x_flatten.get_shape().as_list())
    x_shape = x_flatten.get_shape().as_list()
    in_dim = x_shape[1]

    with tf.variable_scope(name) as scope:
        weights = new_weights('weights', 0, [in_dim, out_dim], initializer=init_w,
                              data_dict=data_dict, trainable=trainable, wd=wd)
        biases = new_biases('biases', 1, [out_dim], initializer=init_b,
                            data_dict=data_dict, trainable=trainable)
        act = tf.nn.xw_plus_b(x_flatten, weights, biases)

        output = nl(act, name='output')
        if re_dict is True:
            return {'outputs': output, 'weights': weights, 'biases': biases}
        else:
            return output

def max_pool(x, name='max_pool', filter_size=2, stride=None, padding='VALID'):
    """ 
    Max pooling layer 

    Args:
        x (tf.tensor): a tensor 
        name (str): name scope of the layer
        filter_size (int or list with length 2): size of filter
        stride (int or list with length 2): Default to be the same as shape
        padding (str): 'VALID' or 'SAME'. Use 'SAME' for FCN.

    Returns:
        tf.tensor with name 'name'
    """

    padding = padding.upper()
    filter_shape = get_shape4D(filter_size)
    if stride is None:
        stride = filter_shape
    else:
        stride = get_shape4D(stride)

    return tf.nn.max_pool(x, ksize=filter_shape, 
                          strides=stride, 
                          padding=padding, name=name)

def global_avg_pool(x, name='global_avg_pool', data_format='NHWC'):
    assert x.shape.ndims == 4
    assert data_format in ['NHWC', 'NCHW']
    with tf.name_scope(name):
        axis = [1, 2] if data_format == 'NHWC' else [2, 3]
        return tf.reduce_mean(x, axis)

# def avg_pool(x, name = 'avg_pool', filter_size = 2, stride = None, padding = 'VALID'):
#     """ 
#     Average pooling layer 

#     Args:
#         x (tf.tensor): a tensor 
#         name (str): name scope of the layer
#         filter_size (int or list with length 2): size of filter
#         stride (int or list with length 2): Default to be the same as shape
#         padding (str): 'VALID' or 'SAME'. Use 'SAME' for FCN.

#     Returns:
#         tf.tensor with name 'name'
#     """

#     padding = padding.upper()
#     filter_shape = get_shape2D(filter_size)
#     if stride is None:
#         stride = filter_shape
#     else:
#         stride = get_shape4D(stride)
#     return tf.nn.pool(x,window_shape = filter_shape, 
#                         pooling_type = 'AVG',
#                         padding = padding,
#                         # strides = stride, 
#                         name = name)

def dropout(x, keep_prob, is_training, name='dropout'):
    """ 
    Dropout 

    Args:
        x (tf.tensor): a tensor 
        keep_prob (float): keep prbability of dropout
        is_training (bool): whether training or not
        name (str): name scope

    Returns:
        tf.tensor with name 'name'
    """

    # tf.nn.dropout does not have 'is_training' argument
    # return tf.nn.dropout(x, keep_prob)
    return tf.layers.dropout(x, rate=1 - keep_prob, 
                            training=is_training, name=name)
    

def batch_norm(x, train=True, name='bn'):
    """ 
    batch normal 

    Args:
        x (tf.tensor): a tensor 
        name (str): name scope
        train (bool): whether training or not

    Returns:
        tf.tensor with name 'name'
    """
    return tf.contrib.layers.batch_norm(x, decay=0.9, 
                          updates_collections=None,
                          epsilon=1e-5, scale=False,
                          is_training=train, scope=name)

def leaky_relu(x, leak=0.2, name='LeakyRelu'):
    """ 
    leaky_relu 
        Allow a small non-zero gradient when the unit is not active

    Args:
        x (tf.tensor): a tensor 
        leak (float): Default to 0.2

    Returns:
        tf.tensor with name 'name'
    """
    return tf.maximum(x, leak*x, name=name)

def new_normal_variable(name, shape=None, trainable=True, stddev=0.002):
    return tf.get_variable(name, shape=shape, trainable=trainable, 
                 initializer=tf.random_normal_initializer(stddev=stddev))

def new_variable(name, idx, shape, initializer=None):
    # initial_value = tf.truncated_normal(shape, 0.0, 0.001)
    # var = tf.get_variable(name, 
    #                        initializer = initial_value)
    # initializer = tf.random_normal_initializer(stddev = 0.002)

    var = tf.get_variable(name, shape=shape, 
                           initializer=initializer) 

    # var_dict[(name, idx)] = var
    return var

def new_weights(name, idx, shape, initializer=None, wd=None, 
                data_dict=None,
                trainable=True): 
    cur_name_scope = tf.get_default_graph().get_name_scope()
    if data_dict is not None and cur_name_scope in data_dict:
        try:
            load_data = data_dict[cur_name_scope][0]
        except KeyError:
            load_data = data_dict[cur_name_scope]['weights']
        print('Load {} weights!'.format(cur_name_scope))

        load_data = np.reshape(load_data, shape)
        initializer = tf.constant_initializer(load_data)
        var = tf.get_variable(name, shape=shape, 
                                  initializer=initializer,
                                  trainable=trainable)
    elif wd is not None:
        print('Random init {} weights with weight decay...'.format(cur_name_scope))
        if initializer is None:
            initializer = tf.truncated_normal_initializer(stddev=0.01)
            # initializer = tf.random_normal_initializer(stddev = 0.002)
        var = tf.get_variable(name, shape=shape, 
                                  initializer=initializer,
                                  trainable=trainable) 
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    else:
        print('Random init {} weights...'.format(cur_name_scope))
        if initializer is None:
            initializer = tf.random_normal_initializer(stddev=0.002)
        var = tf.get_variable(name, shape=shape, 
                            initializer=initializer,
                            trainable=trainable) 
    # var_dict[(name, idx)] = var
    return var

def new_biases(name, idx, shape, initializer=None,
                data_dict=None, trainable=True):
    cur_name_scope = tf.get_default_graph().get_name_scope()
    if data_dict is not None and cur_name_scope in data_dict:
        try:
            load_data = data_dict[cur_name_scope][1]
        except KeyError:
            load_data = data_dict[cur_name_scope]['biases']
        print('Load {} biases!'.format(cur_name_scope))

        load_data = np.reshape(load_data, shape)
        initializer = tf.constant_initializer(load_data)
    else:
        print('Random init {} biases...'.format(cur_name_scope))
        # trainable = True
        if initializer is None:
            initializer = tf.random_normal_initializer(stddev=0.002)
            # initializer = tf.constant_initializer(0)

    var = tf.get_variable(name, shape=shape, 
                           initializer=initializer,
                           trainable=trainable) 
    # var_dict[(name, idx)] = var
    return var



def get_shape2D(in_val):
    """
    Return a 2D shape 

    Args:
        in_val (int or list with length 2) 

    Returns:
        list with length 2
    """
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))

def get_shape4D(in_val):
    """
    Return a 4D shape

    Args:
        in_val (int or list with length 2)

    Returns:
        list with length 4
    """
    # if isinstance(in_val, int):
    return [1] + get_shape2D(in_val) + [1]

def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))

# def variable_with_weight_decay(name, shape, init):
#     var = tf.get_variable(name, shape = shape, 
#                            initializer = initializer)


# From tensorflow tutorial
# def _variable_with_weight_decay(name, shape, stddev, wd):
#   """Helper to create an initialized Variable with weight decay.
#   Note that the Variable is initialized with a truncated normal distribution.
#   A weight decay is added only if one is specified.
#   Args:
#     name: name of the variable
#     shape: list of ints
#     stddev: standard deviation of a truncated Gaussian
#     wd: add L2Loss weight decay multiplied by this float. If None, weight
#         decay is not added for this Variable.
#   Returns:
#     Variable Tensor
#   """
#   # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
#     dtype = tf.float32
#     var = _variable_on_cpu(
#         name,
#         shape,
#         tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
#     if wd is not None:
#         weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
#         tf.add_to_collection('losses', weight_decay)
#     return var


# def _variable_on_cpu(name, shape, initializer):
#   """Helper to create a Variable stored on CPU memory.
#   Args:
#     name: name of the variable
#     shape: list of ints
#     initializer: initializer for Variable
#   Returns:
#     Variable Tensor
#   """
#   # with tf.device('/cpu:0'):
#     dtype = tf.float32
#     var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
#   # return var

