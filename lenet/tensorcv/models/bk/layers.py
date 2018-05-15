import tensorflow as tf

def conv(x, filter_height, filter_width, num_filters, 
         name, stride_x = 1, stride_y = 1, 
         padding = 'SAME', relu = True):
    input_channel = int(x.shape[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k, 
                   strides=[1, stride_y, stride_x, 1], 
                   padding = padding)
    with tf.variable_scope(name) as scope:
        # weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channel, num_filters])
        # biases = tf.get_variable('biases', shape = [num_filters])
        weights = new_normal_variable('weights', 
            shape = [filter_height, filter_width, input_channel, num_filters])
        biases = new_normal_variable('biases', shape = [num_filters])

        conv = convolve(x, weights)
        bias = tf.nn.bias_add(conv, biases)
        # bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        
        if relu:
            relu = tf.nn.relu(bias, name = scope.name)
            return relu
        else:
            return bias

def dconv(x, filter_height, filter_width, 
          name, fuse_x = None, 
          output_shape = [], output_channels = None, 
          stride_x = 2, stride_y = 2, padding = 'SAME'):
    input_channels = int(x.shape[-1])
    
    if fuse_x is not None:
        output_shape = tf.shape(fuse_x)
        output_channels = int(fuse_x.shape[-1])
    elif output_channels is None:
        output_channels = output_shape[-1]

    with tf.variable_scope(name) as scope:
        # weights = tf.get_variable('weights', shape = [filter_height, filter_width, output_channels, input_channels])
        # biases = tf.get_variable('biases', shape = [output_channels])
        weights = new_normal_variable('weights', 
            shape = [filter_height, filter_width, output_channels, input_channels])
        biases = new_normal_variable('biases', shape = [output_channels])

        dconv = tf.nn.conv2d_transpose(x, weights, 
                               output_shape = output_shape, 
                               strides=[1, stride_y, stride_x, 1], 
                               padding = padding, name = scope.name)
        bias = tf.nn.bias_add(dconv, biases)
        bias = tf.reshape(bias, output_shape)

        if fuse_x is not None:
            fuse = tf.add(bias, fuse_x, name = 'fuse')
            return fuse
        else:
            return bias

def fc(x, num_in, num_out, name, relu = True):
    num_in = x.get_shape().as_list()[1]
    # num_in = x.shape[-1]
    with tf.variable_scope(name) as scope:
        # weights = tf.get_variable('weights', shape = [num_in, num_out], trainable = True)
        # biases = tf.get_variable('biases', shape = [num_out], trainable = True)
        weights = new_normal_variable('weights', shape = [num_in, num_out])
        biases = new_normal_variable('biases', shape = [num_out])
        act = tf.nn.xw_plus_b(x, weights, biases, name = scope.name)

        if relu:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

def max_pool(x, name, filter_height = 2, filter_width = 2, 
            stride_x = 2, stride_y = 2, padding = 'SAME'):
    return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1], 
                          strides = [1, stride_y, stride_x, 1], 
                          padding = padding, name = name)

def dropout(x, keep_prob, is_training):
    # print(is_training)
    return tf.layers.dropout(x, rate = 1 - keep_prob, training = is_training)
    # return tf.nn.dropout(x, keep_prob, is_training = is_training)

def batch_norm(x, name, train = True):
    return tf.contrib.layers.batch_norm(x,
                      decay = 0.9, 
                      updates_collections = None,
                      epsilon = 1e-5,
                      scale = False,
                      is_training = train,
                      scope = name)

def leaky_relu(x, leak = 0.2):
    return tf.maximum(x, leak*x)

def new_normal_variable(name, shape = None, trainable = True, stddev = 0.002):
    return tf.get_variable(name, shape = shape, trainable = trainable, 
                 initializer = tf.random_normal_initializer(stddev = stddev))




