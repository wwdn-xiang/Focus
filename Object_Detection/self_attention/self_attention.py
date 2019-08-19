# coding=utf-8
# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   self_attention.py    
@Contact :   384474737@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
19-6-21 上午9:39   alpha      1.0         None
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow.contrib.slim as slim

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope

from tensorflow.python.ops import nn

from tensorflow.python.ops import init_ops

from tensorflow.contrib.layers.python.layers import initializers
import numpy as np

slim = tf.contrib.slim

import functools


def non_local( net, scope=None, n_head=1):
    with tf.variable_scope(scope, default_name="non_local"):
        net0 = cnn_self_attenation(net, 128, n_head)
        return net + net0


def non_local(self, net, scope=None, n_head=1):
    channel = net.get_shape().as_list()[-1]
    conv_op = functools.partial(slim.separable_conv2d, depth_multiplier=1)
    keep_prob = 0.9 if self.use_dropblock else None
    with tf.variable_scope(scope, default_name="non_local"):
        with tf.variable_scope("hattenation_0"):
            net0 = cnn_self_hattenation(net, 128, n_head, keep_prob=keep_prob, is_training=self.is_training)
        with tf.variable_scope("hattenation_1"):
            net0 = cnn_self_hattenation(net0, 128, n_head, keep_prob=keep_prob, is_training=self.is_training)
        with tf.variable_scope("vattenation_0"):
            net1 = cnn_self_vattenation(net, 128, n_head, keep_prob=keep_prob, is_training=self.is_training)
        with tf.variable_scope("vattenation_1"):
            net1 = cnn_self_vattenation(net1, 128, n_head, keep_prob=keep_prob, is_training=self.is_training)
        net_f = tf.concat([net0, net1, net], axis=-1)
        with tf.variable_scope("select"):
            score = conv_op(net_f, 1, [3, 3], activation_fn=tf.nn.sigmoid, rate=2)
            tf.add_to_collection("WDEBUG", score)
        with tf.variable_scope("fusion"):
            net_o = tf.concat([score * net0, (1.0 - score) * net1], axis=-1)
            net_o = conv_op(net_o, channel, [3, 3])
            net_o = tf.concat([net_o, net], axis=-1)
            net_o = conv_op(net_o, channel, [3, 3])
        return net_o


def cnn_self_attenation(net, channel=None, n_head=1, keep_prob=None, is_training=False):
    old_channel = net.get_shape().as_list()[-1]
    if channel is not None:
        net = slim.conv2d(net, channel, [1, 1], scope="projection_0")

    shape = net.get_shape().as_list()

    new_shape = [-1, shape[1] * shape[2], shape[3]]

    net = tf.reshape(net, new_shape)

    net = self_attenation(net, n_head=n_head, keep_prob=keep_prob, is_training=is_training, use_mask=False)

    if channel is not None:

        out_shape = [-1, shape[1], shape[2], channel]

        net = tf.reshape(net, out_shape)

        net = slim.conv2d(net, old_channel, [1, 1], scope="projection_1")

    else:

        out_shape = [-1, shape[1], shape[2], old_channel]

        net = tf.reshape(net, out_shape)

    return net


def self_attenation(net, *args, **kwargs):
    with tf.variable_scope("SelfAttention"):
        channel = net.get_shape().as_list()[-1]

        net = tf.layers.dense(

            net,

            channel * 3,

            activation=None,

            name="split_to_QKV",

            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        Q, K, V = tf.split(net, num_or_size_splits=3, axis=2)

        return multi_head_attention(Q, K, V, *args, **kwargs)


'''

生成一个mask，用于保证当前的输出只依赖于之前的数据，主要用于decoder

'''


def mask_attn_weights(w):
    n = w.get_shape().as_list()[-1]

    # 取下三角

    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)

    b = tf.reshape(b, [1, 1, n, n])

    w = w * b + -1e9 * (1 - b)

    return w


'''

inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；

seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；

mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；

add是指把多出部分全部减去一个大的常数，一般用于softmax之前。



用于配合length使用，防止attention到有效数据之外的数据

'''


def mask_attn_of_length(inputs, seq_len, mode='mul'):
    if seq_len == None:

        return inputs

    else:

        max_seq_len = inputs.get_shape().as_list()[1]

        if max_seq_len is None:
            max_seq_len = tf.shape(inputs)[1]

        mask = tf.cast(tf.sequence_mask(seq_len, maxlen=max_seq_len), tf.float32)

        for _ in range(len(inputs.shape) - 2):
            mask = tf.expand_dims(mask, 2)

        if mode == 'mul':
            return inputs * mask

        if mode == 'add':
            return inputs - (1 - mask) * 1e12


'''

Multi-Head Attention from ATTENTION IS ALL YOU NEED

Q:[batch_size,n,dk]

K:[batch_size,m,dk]

V:[batch_size,m,dv]

use_mask: where use the mask, in the original paper，use mask for decode, don't use mask for encode

V_len is important, ensure softmax(A) attention on valid value, Q_len donsen't that important

if use multi layer of attention every layer have to use V_len

res:[batch_size,n,nb_head*size_per_head]

'''


def multi_head_attention(Q, K, V, n_head, Q_len=None, V_len=None, keep_prob=None, is_training=False, use_mask=False):
    with tf.variable_scope("MultiHeadAttention"):

        Q = split_heads(Q, n_head)

        # now Q is [batch_size, nb_head,n,size_per_head0]

        K = split_heads(K, n_head)

        # now K is [batch_size,nb_head,m,size_per_head0]

        V = split_heads(V, n_head)

        # new V is [batch_size,nb_head,m,size_per_head1]

        # 计算内积，然后mask，然后softmax

        A = tf.matmul(Q, K, transpose_b=True, name="MatMul_QK") * tf.rsqrt(tf.cast(tf.shape(V)[-1], tf.float32))

        # now A is [batch_size, nb_head,n,m]

        if use_mask:
            A = mask_attn_weights(A)

        if V_len is not None:
            A = tf.transpose(A, [0, 3, 2, 1])

            A = mask_attn_of_length(A, V_len, mode='add')

            A = tf.transpose(A, [0, 3, 2, 1])

        A = tf.nn.softmax(A)

        if keep_prob is not None:
            A = slim.dropout(A, keep_prob=keep_prob, is_training=is_training)

        # 输出并mask

        O = tf.matmul(A, V, name="MatMul_AV")

        # 如果使用mask,O的第i行为V的第0到第i行的线性加权组合, 否则为所有的加权

        # now O is  [batch_size,nb_head,n,size_per_head1]

        O = tf.transpose(O, [0, 2, 1, 3])

        # now O is  [batch_size,n,nb_head,size_per_head1]

        o_shape = O.get_shape().as_list()

        if o_shape[0] is not None:

            O = tf.reshape(O, [o_shape[0], -1, np.prod(o_shape[-2:])])

        else:

            O = tf.reshape(O, [-1, o_shape[1], np.prod(o_shape[-2:])])

        if Q_len is not None:
            O = mask_attn_of_length(O, Q_len, 'mul')

        return O


def split_heads(x, n):
    '''

    :param x:  [batch_size,M,D]

    :param n:  分割数

    :return: [batch_size,n,M,D//n]

    '''

    return tf.transpose(split_states(x, n), [0, 2, 1, 3])


def reshape(tensor, shape, name=None):
    if isinstance(shape, list):
        shape = [x if (isinstance(x, tf.Tensor) or (x is not None and x >= 0)) else -1 for x in shape]

        return tf.reshape(tensor, shape, name)

    return tf.reshape(tensor, shape, name)


def split_states(x, n):
    '''

    :param x:  [batch_size,M,D]

    :param n:  分割数

    :return: [batch_size,M,n,D//n]

    '''

    x_shape = x.get_shape().as_list()

    m = x_shape[-1]

    new_x_shape = x_shape[:-1] + [n, m // n]

    return reshape(x, new_x_shape)


def v_operation(inputs, activation_fn=nn.relu):
    '''

    V operation from CONVLUTION SEQUENCE TO SEQUENCE LEARNING

    :param inputs: [batch_size,len,d]

    :param activation_fn:

    :return: [batch_size,len,d/2]

    '''

    d = inputs.get_shape().as_list()[2]

    A = inputs[:, :, :d / 2]

    B = inputs[:, :, d / 2:]

    V = tf.multiply(A, activation_fn(B))

    return V


def highway(x, size=None, activation_fn=None,

            num_layers=2, scope="highway", keep_prob=None, reuse=None):
    '''



    :param x: [batch_size,W,C]

    :param size:

    :param activation:

    :param num_layers:

    :param scope:

    :param dropout:

    :param reuse:

    :return:

    '''

    with tf.variable_scope(scope, reuse):

        if size is None:

            size = x.shape.as_list()[-1]

        else:

            x = slim.convolution1d(x, size, [1], scope="input_projection", reuse=reuse, biases_initializer=None)

        for i in range(num_layers):

            T = slim.convolution1d(x, size, [1], activation_fn=tf.sigmoid,

                                   scope="gate_%d" % i, reuse=reuse)

            H = slim.convolution1d(x, size, [1], activation_fn=activation_fn,

                                   scope="activation_%d" % i, reuse=reuse)

            if keep_prob is not None:
                H = slim.dropout(H, keep_prob)

            x = H * T + x * (1.0 - T)

        return x


def cnn_self_hattenation(net, channel=None, n_head=1, keep_prob=None, is_training=False):
    old_channel = net.get_shape().as_list()[-1]

    if channel is not None:
        net = slim.conv2d(net, channel, [1, 1], scope="projection_0")

    shape = net.get_shape().as_list()

    new_shape = [-1, shape[2], shape[3]]

    net = tf.reshape(net, new_shape)

    net = self_attenation(net, n_head=n_head, keep_prob=keep_prob, is_training=is_training, use_mask=False)

    if channel is not None:

        out_shape = [-1, shape[1], shape[2], channel]

        net = tf.reshape(net, out_shape)

        net = slim.conv2d(net, old_channel, [1, 1], scope="projection_1")

    else:

        out_shape = [-1, shape[1], shape[2], old_channel]

        net = tf.reshape(net, out_shape)

    return net


def cnn_self_vattenation(net, channel=None, n_head=1, keep_prob=None, is_training=False):
    net = tf.transpose(net, perm=[0, 2, 1, 3], name="transpose_0")

    output = cnn_self_hattenation(net, channel, n_head, keep_prob, is_training=is_training)

    return tf.transpose(output, perm=[0, 2, 1, 3], name="transpose_1")


def self_attenation(net, *args, **kwargs):
    with tf.variable_scope("SelfAttention"):
        channel = net.get_shape().as_list()[-1]

        net = tf.layers.dense(

            net,

            channel * 3,

            activation=None,

            name="split_to_QKV",

            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        Q, K, V = tf.split(net, num_or_size_splits=3, axis=2)

        return multi_head_attention(Q, K, V, *args, **kwargs)


'''

Multi-Head Attention from ATTENTION IS ALL YOU NEED

Q:[batch_size,n,dk]

K:[batch_size,m,dk]

V:[batch_size,m,dv]

use_mask: where use the mask, in the original paper，use mask for decode, don't use mask for encode

V_len is important, ensure softmax(A) attention on valid value, Q_len donsen't that important

if use multi layer of attention every layer have to use V_len

res:[batch_size,n,nb_head*size_per_head]

'''


def multi_head_attention(Q, K, V, n_head, Q_len=None, V_len=None, keep_prob=None, is_training=False, use_mask=False):
    with tf.variable_scope("MultiHeadAttention"):

        Q = split_heads(Q, n_head)

        # now Q is [batch_size, nb_head,n,size_per_head0]

        K = split_heads(K, n_head)

        # now K is [batch_size,nb_head,m,size_per_head0]

        V = split_heads(V, n_head)

        # new V is [batch_size,nb_head,m,size_per_head1]

        # 计算内积，然后mask，然后softmax

        A = tf.matmul(Q, K, transpose_b=True, name="MatMul_QK") * tf.rsqrt(tf.cast(tf.shape(V)[-1], tf.float32))

        # now A is [batch_size, nb_head,n,m]

        if use_mask:
            A = mask_attn_weights(A)

        if V_len is not None:
            A = tf.transpose(A, [0, 3, 2, 1])

            A = mask_attn_of_length(A, V_len, mode='add')

            A = tf.transpose(A, [0, 3, 2, 1])

        A = tf.nn.softmax(A)

        if keep_prob is not None:
            A = slim.dropout(A, keep_prob=keep_prob, is_training=is_training)

        # 输出并mask

        O = tf.matmul(A, V, name="MatMul_AV")

        # 如果使用mask,O的第i行为V的第0到第i行的线性加权组合, 否则为所有的加权

        # now O is  [batch_size,nb_head,n,size_per_head1]

        O = tf.transpose(O, [0, 2, 1, 3])

        # now O is  [batch_size,n,nb_head,size_per_head1]

        o_shape = O.get_shape().as_list()

        if o_shape[0] is not None:

            O = tf.reshape(O, [o_shape[0], -1, np.prod(o_shape[-2:])])

        else:

            O = tf.reshape(O, [-1, o_shape[1], np.prod(o_shape[-2:])])

        if Q_len is not None:
            O = mask_attn_of_length(O, Q_len, 'mul')

        return O


if __name__ == '__main__':
    net = tf.placeholder(shape=[None, 6, 6, 8], dtype=tf.float32)
    data = np.arange(6 * 6 * 8).reshape([1, 6, 6, 8])
    data = data.astype(np.float32)
    output = cnn_self_attenation(net,128)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(output, feed_dict={net: data}).shape)
