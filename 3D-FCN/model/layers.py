import tensorflow as tf

def fully_connected(input_layer, shape, name="", is_training=True):
    with tf.variable_scope("fully" + name):
        kernel = tf.get_variable("weights", shape=shape, \
            dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        fully = tf.matmul(input_layer, kernel)
        fully = tf.nn.relu(fully)
        fully = batch_norm(fully, is_training)
        return fully

def max_pool(input_layer, ksizes, stride, name="", padding='SAME'):
    with tf.variable_scope("Pool3D"+name):
        pool =tf.nn.max_pool3d(input=input_layer, ksize=ksizes, strides=stride, padding=padding, name=name)
    return pool

def average_pool(input_layer, ksizes, stride, name="", padding='SAME'):
    with tf.variable_scope("average3d"+name):
        avg =tf.nn.avg_pool3d(input=input_layer,ksize=ksizes,strides=stride,padding=padding,name=name)
    return avg

def conv3d_layer(input, filter, kernel, stride, use_bias, activation, name="CONV3d"):
    with tf.name_scope(name):
        network =tf.layers.conv3d(inputs=input,filters=filter,kernel_size=kernel, strides=stride, 
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), bias_initializer=tf.zeros_initializer(),
            use_bias=use_bias, activation = activation, padding="SAME",name=name)
        return network         

def conv3d_transpose_layer(input, filter, kernel, stride, use_bias, activation, name="DECONV3d"):
    with tf.name_scope(name):
        network =tf.layers.conv3d_transpose(inputs=input,filters=filter,kernel_size=kernel, strides=stride, 
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), bias_initializer=tf.zeros_initializer(),
            use_bias=use_bias, activation = activation, padding="SAME",name=name)
        return network          

def batch_norm(inputs, is_training, decay=0.9, eps=1e-5,name=''):
    """Batch Normalization

       Args:
           inputs: input data(Batch size) from last layer
           is_training: True if train phase, None if test phase
       Returns:
           output for next layer
    """
    with tf.variable_scope('BN'+name):
        gamma = tf.get_variable("gamma", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        pop_mean = tf.get_variable("pop_mean", trainable=False, shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        pop_var = tf.get_variable("pop_var", trainable=False, shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        axes = list(range(len(inputs.get_shape()) - 1))

        if is_training is not None:
            batch_mean, batch_var = tf.nn.moments(inputs, axes)
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean*(1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, eps)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, eps)

# def conv3DLayer(input_layer, input_dim, output_dim, height, width, length, stride, activation=tf.nn.relu, padding="SAME", name="", is_training=True):
#     with tf.variable_scope("conv3D" + name):
#         kernel = tf.get_variable("weights", shape=[length, height, width, input_dim, output_dim], \
#             dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
#         b = tf.get_variable("bias", shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
#         conv = tf.nn.conv3d(input_layer, kernel, stride, padding=padding)
#         bias = tf.nn.bias_add(conv, b)
#         if activation:
#             bias = activation(bias, name="activation")
#         bias = batch_norm(bias, is_training)
#     return bias
# def conv3D_to_output(input_layer, input_dim, output_dim, height, width, length, stride, activation=tf.nn.relu, padding="SAME", name=""):
#     with tf.variable_scope("conv3D" + name):
#         kernel = tf.get_variable("weights", shape=[length, height, width, input_dim, output_dim], \
#             dtype=tf.float32, initializer=tf.constant_initializer(0.01))
#         conv = tf.nn.conv3d(input_layer, kernel, stride, padding=padding)
#     return conv
# def deconv3D_to_output(input_layer, input_dim, output_dim, height, width, length, stride, output_shape, activation=tf.nn.relu, padding="SAME", name=""):
#     with tf.variable_scope("deconv3D"+name):
#         kernel = tf.get_variable("weights", shape=[length, height, width, output_dim, input_dim], \
#             dtype=tf.float32, initializer=tf.constant_initializer(0.01))
#         deconv = tf.nn.conv3d_transpose(input_layer, kernel, output_shape, stride, padding="SAME")
#     return deconv
# def conv_layer(input, filter, kernel, stride=1, name="CONV"):
#     with tf.name_scope(layer_name):
#         network =tf.layers.conv3d(inputs=input,use_bias=False,filters=filter,kernel_size=kernel,strides=stride,padding="SAME",name=layer_name+"c")
#         return network  
# def Drop_out(input_layer, rate,training):
#     return tf.layers.dropout(inputs=input_layer,rate=rate,training=training)
# def Relu(x):
#     return tf.nn.relu(x)        
# def Concatenation(layers):
#     return tf.concat(layers,axis=4)