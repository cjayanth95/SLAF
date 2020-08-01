from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_variables(name, shape, initializer=tf.initializers.random_normal(), scale=None):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = None
    if scale is not None:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.00005)

    new_variables = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

def bias_variable(name, shape, trainable=True):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.get_variable(name=name, dtype=tf.float32, trainable=trainable, initializer=initial,
                                    regularizer=None)


def update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay):
    update_mean = tf.assign(pop_mean, batch_mean*(1-decay) + pop_mean*(decay))
    update_var = tf.assign(pop_var, batch_var*(1-decay) + pop_var*(decay))
    with tf.control_dependencies([update_mean, update_var]):
        temp = tf.nn.batch_normalization(X, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=1e-8)
        return temp


def legendre(inp, k, is_train=False, name="activation", scale=None, decay=0.9, reuse=False):

    X = inp

    with tf.variable_scope(name, reuse=reuse):

        if len(X.get_shape()) == 4:


            (batch, H, W, C) = X.get_shape()

            shape = X.get_shape().as_list()
            rank = len(shape)
            input_r = X
            l = []
            temp = 1.0
            # for i in range(k):
            #     temp = temp * input_r
            #     print("templine56")
            #     print(temp)
            #     l.append(temp)
            temp1 = tf.pow(input_r, 0)
            tempx = tf.pow(input_r, 1)
            tempx2 = tf.pow(input_r, 2)
            tempx3 = tf.pow(input_r, 3)
            tempx4 = tf.pow(input_r, 4)
            tempx5 = tf.pow(input_r, 5)
            tempc0_5 = tf.constant(0.5, name="scalar_a", dtype=tf.float32)
            tempc5 = tf.constant(5, name="scalar_a", dtype=tf.float32)
            tempc3 = tf.constant(3, name="scalar_b", dtype=tf.float32)
            temp5x3 = tf.multiply(tf.pow(input_r,3),tempc5,'multiply5')
            temp3x = tf.multiply(tf.pow(input_r,1),tempc3,'multiply3')
            templ3 = tf.multiply(tf.multiply(tf.pow(input_r,2),tempc3,'multiply3')-temp1,tempc0_5,'multiply0.5')
            templ4 = tf.multiply(temp5x3-temp3x,tempc0_5,'multiply0.5')
            tempc35 = tf.constant(35, name="scalar_a", dtype=tf.float32)
            tempc30 = tf.constant(30, name="scalar_a", dtype=tf.float32)
            temp35x4 = tf.multiply(tf.pow(input_r,4),tempc35,'multiply35')
            temp30x2 = tf.multiply(tf.pow(input_r,2),tempc30,'multiply30')
            tempc0125 = tf.constant(0.125, name="scalar_a", dtype=tf.float32)
            templ5 = tf.multiply(temp35x4-temp30x2+tempc3,tempc0125,'multiply0.125')
            tempc63 = tf.constant(63, name="scalar_a", dtype=tf.float32)
            tempc70 = tf.constant(70, name="scalar_a", dtype=tf.float32)
            tempc15 = tf.constant(15, name="scalar_a", dtype=tf.float32)
            temp63x5 = tf.multiply(tf.pow(input_r,5),tempc63,'multiply63')
            temp70x3 = tf.multiply(tf.pow(input_r,5),tempc70,'multiply70')
            temp15x = tf.multiply(tf.pow(input_r,1),tempc15,'multiply15')
            templ6 = tf.multiply(temp63x5-temp70x3 + temp15x,tempc0125,'multiply0.125')
            #l.append(temp1)
            l.append(tempx)
            #l.append(tempx2)
            # l.append(tempx3)
            # l.append(tempx4)
            l.append(templ4)
            print("templ3")
            # print(templ3)
            # l.append(templ4)
            l.append(templ6)
            if(k == 5):
                l.append(templ5)
            X = tf.stack(l, axis=(rank))

            batch_mean, batch_var = tf.nn.moments(X, axes=[0, 1, 2], keep_dims=True)
            pop_mean = tf.get_variable(initializer=tf.zeros([1, 1, 1, C, k]), trainable = False, name="activation_means")
            pop_var = tf.get_variable(initializer=tf.ones([1, 1, 1, C, k]), trainable = False, name="activation_variances")
            # max_tensor = tf.cond(is_train, lambda: update_op(pop_max, max_tensor, m, n, decay), lambda: m*max_tensor + n)
            if type(is_train) is not bool:
                Xin = tf.cond(is_train, lambda: update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay), lambda: tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8))
            else:
                if is_train:
                    Xin = update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay)
                else:
                    Xin = tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8)
            
            attention = tf.nn.sigmoid(create_variables("activation_attention", [1, k]))
            kernel = create_variables("activation_weights", [C, k], scale=scale) + tf.concat([tf.ones([C, 1]), tf.zeros([C, k-1])], axis=-1) 
            coeff = tf.identity(attention * kernel, name="activation_coeff")
            bias = bias_variable(name="activation_bias", shape=[C])
            print(bias)
            conv = tf.reduce_sum(attention * kernel * Xin, axis=-1) + bias
            return(conv)
        if len(X.get_shape()) == 3:
            (batch, W, C) = X.get_shape()
            shape = X.get_shape().as_list()
            rank = len(shape)
            input_r = X
            l = []
            temp = 1.0
            # for i in range(k):
            #     temp = temp * input_r
            #     print("templine93")
            #     print(temp)
            #     l.append(temp)
            temp1 = tf.pow(input_r, 0)
            tempx = tf.pow(input_r, 1)
            tempx2 = tf.pow(input_r, 2)
            tempx3 = tf.pow(input_r, 3)
            tempx4 = tf.pow(input_r, 4)
            tempx5 = tf.pow(input_r, 5)
            tempc0_5 = tf.constant(0.5, name="scalar_a", dtype=tf.float32)
            tempc5 = tf.constant(5, name="scalar_a", dtype=tf.float32)
            tempc3 = tf.constant(3, name="scalar_b", dtype=tf.float32)
            temp5x3 = tf.multiply(tf.pow(input_r,3),tempc5,'multiply5')
            temp3x = tf.multiply(tf.pow(input_r,1),tempc3,'multiply3')
            templ3 = tf.multiply(tf.multiply(tf.pow(input_r,2),tempc3,'multiply3')-temp1 ,tempc0_5,'multiply0.5')
            templ4 = tf.multiply(temp5x3-temp3x,tempc0_5,'multiply0.5')
            tempc35 = tf.constant(35, name="scalar_a", dtype=tf.float32)
            tempc30 = tf.constant(30, name="scalar_a", dtype=tf.float32)
            temp35x4 = tf.multiply(tf.pow(input_r,4),tempc35,'multiply35')
            temp30x2 = tf.multiply(tf.pow(input_r,2),tempc30,'multiply30')
            tempc0125 = tf.constant(0.125, name="scalar_a", dtype=tf.float32)
            templ5 = tf.multiply(temp35x4-temp30x2+tempc3,tempc0125,'multiply0.125')
            tempc63 = tf.constant(63, name="scalar_a", dtype=tf.float32)
            tempc70 = tf.constant(70, name="scalar_a", dtype=tf.float32)
            tempc15 = tf.constant(15, name="scalar_a", dtype=tf.float32)
            temp63x5 = tf.multiply(tf.pow(input_r,5),tempc63,'multiply63')
            temp70x3 = tf.multiply(tf.pow(input_r,5),tempc70,'multiply70')
            temp15x = tf.multiply(tf.pow(input_r,1),tempc15,'multiply15')
            templ6 = tf.multiply(temp63x5-temp70x3 + temp15x,tempc0125,'multiply0.125')
            l.append(tempx)
            # l.append(tempx2)
            # l.append(tempx3)
            #l.append(tempx4)
            #l.append(templ3)
            print("templ3")
            # print(templ3)
            l.append(templ4)
            l.append(templ6)
            if(k == 5):
                l.append(templ5)
            X = tf.stack(l, axis=(rank))

            batch_mean, batch_var = tf.nn.moments(X, axes=[0, 1], keep_dims=True)
            pop_mean = tf.get_variable(initializer=tf.zeros([1, 1, C, k]), trainable = False, name="activation_means")
            pop_var = tf.get_variable(initializer=tf.ones([1, 1, C, k]), trainable = False, name="activation_variances")
            # max_tensor = tf.cond(is_train, lambda: update_op(pop_max, max_tensor, m, n, decay), lambda: m*max_tensor + n)
            if type(is_train) is not bool:
                Xin = tf.cond(is_train, lambda: update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay), lambda: tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8))
            else:
                if is_train:
                    Xin = update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay)
                else:
                    Xin = tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8)

            attention = tf.nn.sigmoid(create_variables("activation_attention", [1, k]))
            kernel = create_variables("activation_weights", [C, k], scale=scale) + tf.concat([tf.ones([C, 1]), tf.zeros([C, k-1])], axis=-1) 
            coeff = tf.identity(attention * kernel, name="activation_coeff")
            
            bias = bias_variable(name="activation_bias", shape=[C])
            conv = tf.reduce_sum(attention * kernel * Xin, axis=-1) + bias
            return(conv)

        if len(X.get_shape()) == 2:
            (batch, N) = X.get_shape()

            shape = X.get_shape().as_list()
            rank = len(shape)
            input_r = X
            l = []
            temp = 1.0
            # for i in range(k):
            #     temp = temp * input_r
            #     print("templine126")
            #     print(temp)
            #     l.append(temp)
            temp1 = tf.pow(input_r, 0)
            tempx = tf.pow(input_r, 1)
            tempx2 = tf.pow(input_r, 2)
            tempx3 = tf.pow(input_r, 3)
            tempx4 = tf.pow(input_r, 4)
            tempx5 = tf.pow(input_r, 5)
            tempc0_5 = tf.constant(0.5, name="scalar_a", dtype=tf.float32)
            tempc5 = tf.constant(5, name="scalar_a", dtype=tf.float32)
            tempc3 = tf.constant(3, name="scalar_b", dtype=tf.float32)
            temp5x3 = tf.multiply(tf.pow(input_r,3),tempc5,'multiply5')
            temp3x = tf.multiply(tf.pow(input_r,1),tempc3,'multiply3')
            templ3 = tf.multiply(tf.multiply(tf.pow(input_r,2),tempc3,'multiply3')-temp1,tempc0_5,'multiply0.5')
            templ4 = tf.multiply(temp5x3-temp3x,tempc0_5,'multiply0.5')
            tempc35 = tf.constant(35, name="scalar_a", dtype=tf.float32)
            tempc30 = tf.constant(30, name="scalar_a", dtype=tf.float32)
            temp35x4 = tf.multiply(tf.pow(input_r,4),tempc35,'multiply35')
            temp30x2 = tf.multiply(tf.pow(input_r,2),tempc30,'multiply30')
            tempc0125 = tf.constant(0.125, name="scalar_a", dtype=tf.float32)
            templ5 = tf.multiply(temp35x4-temp30x2+tempc3,tempc0125,'multiply0.125')
            tempc63 = tf.constant(63, name="scalar_a", dtype=tf.float32)
            tempc70 = tf.constant(70, name="scalar_a", dtype=tf.float32)
            tempc15 = tf.constant(15, name="scalar_a", dtype=tf.float32)
            temp63x5 = tf.multiply(tf.pow(input_r,5),tempc63,'multiply63')
            temp70x3 = tf.multiply(tf.pow(input_r,5),tempc70,'multiply70')
            temp15x = tf.multiply(tf.pow(input_r,1),tempc15,'multiply15')
            templ6 = tf.multiply(temp63x5-temp70x3 + temp15x,tempc0125,'multiply0.125')
            l.append(tempx)
            # l.append(tempx2)
            # l.append(tempx3)
            # l.append(tempx4)
            #l.append(templ3)
            print("templ3")
            # print(templ3)
            l.append(templ4)
            l.append(templ6)
            if(k == 5):
                l.append(templ5)
            X = tf.stack(l, axis=(rank))

            batch_mean, batch_var = tf.nn.moments(X, axes=[0], keep_dims=True)
            pop_mean = tf.get_variable(initializer=tf.zeros([1, N, k]), trainable = False, name="activation_means")
            pop_var = tf.get_variable(initializer=tf.ones([1, N, k]), trainable = False, name="activation_variances")
            # max_tensor = tf.cond(is_train, lambda: update_op(pop_max, max_tensor, m, n, decay), lambda: m*max_tensor + n)
            if type(is_train) is not bool:
                Xin = tf.cond(is_train, lambda: update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay), lambda: tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8))
            else:
                if is_train:
                    Xin = update_op(X, pop_mean, pop_var, batch_mean, batch_var, decay)
                else:
                    Xin = tf.nn.batch_normalization(X, pop_mean, pop_var, offset=None, scale=None, variance_epsilon=1e-8)

            attention = tf.nn.sigmoid(create_variables("activation_attention", [1, k]))
            kernel = create_variables("activation_weights", [1, k]) + tf.concat([tf.ones([1, 1]), tf.zeros([1, k-1])], axis=-1) 
            coeff = tf.identity(attention * kernel, name="activation_coeff")

            bias = bias_variable(name="activation_bias", shape=[1])
            conv = tf.reduce_sum(attention * kernel * Xin, axis=-1) + bias
            return(conv)