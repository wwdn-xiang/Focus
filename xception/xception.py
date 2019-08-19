#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   xception.py
@Contact :   384474737@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
19-7-17 下午5:13   alpha      1.0         None
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range

import tensorflow as tf

slim = tf.contrib.slim

'''
==================================================================
Based on the Xception Paper (https://arxiv.org/pdf/1610.02357.pdf)
==================================================================

REGULARIZATION CONFIGURATION:
- weight_decay: 1e-5
- dropout: no dropout
- aux_loss: no aux loss

OPTIMIZATION CONFIGURATION (for Google JFT Dataset):
- optimizer: RMSProp
- momentum: 0.9
- initial_learning_rate: 0.001
- learning_rate_decay: 0.9 every 3/350 epochs (every 3M images; total 350M images per epoch)

'''


def xception(inputs,
             num_classes=1001,
             is_training=True,
             scope='xception'):
    '''
    The Xception Model!

    Note:
    The padding is included by default in slim.conv2d to preserve spatial dimensions.

    INPUTS:
    - inputs(Tensor): a 4D Tensor input of shape [batch_size, height, width, num_channels]
    - num_classes(int): the number of classes to predict
    - is_training(bool): Whether or not to train

    OUTPUTS:
    - logits (Tensor): raw, unactivated outputs of the final layer
    - end_points(dict): dictionary containing the outputs for each layer, including the 'Predictions'
                        containing the probabilities of each output.

    :return:
                - logits (Tensor): raw, unactivated outputs of the final layer
                - end_points(dict): dictionary containing the outputs for each layer, including the 'Predictions'
                  containing the probabilities of each output.
    '''
    '''
    # 当reuse为False或者None时（这也是默认值），同一个tf.variable_scope下面的变量名不能相同；当reuse为True时，tf.variable_scope只能获取已经创建过的变量。
    # 简而言之，reuse=False时，tf.variable_scope创建变量；reuse=True时，tf.variable_scope获取变量。

  
    除了tf.variable_scope，tf.name_scope函数也提供了命名空间管理的功能。这两个函数在大部分情况下是等价的，唯一的区别是在使用tf.get_variable函数时。 
    tf.get_variable函数不受tf.name_scope的影响。 
    我们从代码看下这句话的具体意思。 
            首先是tf.variable_scope：
            with tf.variable_scope('foo'):
                a = tf.get_variable('bar',[1])
                print(a.name)#结果为foo/bar:0
            　　
            再看tf.name_scope：
            with tf.name_scope('a'):
                a=tf.Variable([1])
                print(a.name)#结果为a/Variable:0
             
                b=tf.get_variable('b',[1])
                print(b.name)#结果为b:0
　　
 
        从这个结果中，我们能很清晰地看到，tf.get_variable创建的变量并不是a/b:0，而是b:0。这就表示了在tf.name_scope函数下，tf.get_variable不受其约束。
   
    with tf.variable_scope("Xception") as sc:
        end_points_collection = sc.name + "_end_points"
        
        使用slim.arg_scope（）有两个步骤:
            1. 使用@slim.add_arg_scope修饰目标函数
            2. 用 slim.arg_scope（）为目标函数设置默认参数.
        平常所用到的slim.conv2d( ),slim.fully_connected( ),slim.max_pool2d( )等函数在他被定义的时候就已经添加了@add_arg_scope
    
    '''
    with tf.variable_scope('Xception') as sc:
        end_points_collection = sc.name + '_end_points'

        with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1), \
             slim.arg_scope([slim.separable_conv2d, slim.conv2d, slim.avg_pool2d],
                            outputs_collections=[end_points_collection]), \
             slim.arg_scope([slim.batch_norm], is_training=is_training):
            # outputs_collections 指定输出被添加的集合
            # ===========ENTRY FLOW==============
            # Block 1
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='valid', scope='block1_conv1')
            net = slim.batch_norm(net, scope='block1_bn1')
            net = tf.nn.relu(net, name='block1_relu1')
            net = slim.conv2d(net, 64, [3, 3], padding='valid', scope='block1_conv2')
            net = slim.batch_norm(net, scope='block1_bn2')
            net = tf.nn.relu(net, name='block1_relu2')
            residual = slim.conv2d(net, 128, [1, 1], stride=2, scope='block1_res_conv')
            residual = slim.batch_norm(residual, scope='block1_res_bn')

            # Block 2
            net = slim.separable_conv2d(net, 128, [3, 3], scope='block2_dws_conv1')
            net = slim.batch_norm(net, scope='block2_bn1')
            net = tf.nn.relu(net, name='block2_relu1')
            net = slim.separable_conv2d(net, 128, [3, 3], scope='block2_dws_conv2')
            net = slim.batch_norm(net, scope='block2_bn2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block2_max_pool')
            net = tf.add(net, residual, name='block2_add')
            residual = slim.conv2d(net, 256, [1, 1], stride=2, scope='block2_res_conv')
            residual = slim.batch_norm(residual, scope='block2_res_bn')

            # Block 3
            net = tf.nn.relu(net, name='block3_relu1')
            net = slim.separable_conv2d(net, 256, [3, 3], scope='block3_dws_conv1')
            net = slim.batch_norm(net, scope='block3_bn1')
            net = tf.nn.relu(net, name='block3_relu2')
            net = slim.separable_conv2d(net, 256, [3, 3], scope='block3_dws_conv2')
            net = slim.batch_norm(net, scope='block3_bn2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block3_max_pool')
            net = tf.add(net, residual, name='block3_add')
            residual = slim.conv2d(net, 728, [1, 1], stride=2, scope='block3_res_conv')
            residual = slim.batch_norm(residual, scope='block3_res_bn')

            # Block 4
            net = tf.nn.relu(net, name='block4_relu1')
            net = slim.separable_conv2d(net, 728, [3, 3], scope='block4_dws_conv1')
            net = slim.batch_norm(net, scope='block4_bn1')
            net = tf.nn.relu(net, name='block4_relu2')
            net = slim.separable_conv2d(net, 728, [3, 3], scope='block4_dws_conv2')
            net = slim.batch_norm(net, scope='block4_bn2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block4_max_pool')
            net = tf.add(net, residual, name='block4_add')

            # ===========MIDDLE FLOW===============
            for i in range(8):
                block_prefix = 'block%s_' % (str(i + 5))

                residual = net
                net = tf.nn.relu(net, name=block_prefix + 'relu1')
                net = slim.separable_conv2d(net, 728, [3, 3], scope=block_prefix + 'dws_conv1')
                net = slim.batch_norm(net, scope=block_prefix + 'bn1')
                net = tf.nn.relu(net, name=block_prefix + 'relu2')
                net = slim.separable_conv2d(net, 728, [3, 3], scope=block_prefix + 'dws_conv2')
                net = slim.batch_norm(net, scope=block_prefix + 'bn2')
                net = tf.nn.relu(net, name=block_prefix + 'relu3')
                net = slim.separable_conv2d(net, 728, [3, 3], scope=block_prefix + 'dws_conv3')
                net = slim.batch_norm(net, scope=block_prefix + 'bn3')
                net = tf.add(net, residual, name=block_prefix + 'add')

            # ========EXIT FLOW============
            residual = slim.conv2d(net, 1024, [1, 1], stride=2, scope='block12_res_conv')
            residual = slim.batch_norm(residual, scope='block12_res_bn')
            net = tf.nn.relu(net, name='block13_relu1')
            net = slim.separable_conv2d(net, 728, [3, 3], scope='block13_dws_conv1')
            net = slim.batch_norm(net, scope='block13_bn1')
            net = tf.nn.relu(net, name='block13_relu2')
            net = slim.separable_conv2d(net, 1024, [3, 3], scope='block13_dws_conv2')
            net = slim.batch_norm(net, scope='block13_bn2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block13_max_pool')
            net = tf.add(net, residual, name='block13_add')

            net = slim.separable_conv2d(net, 1536, [3, 3], scope='block14_dws_conv1')
            net = slim.batch_norm(net, scope='block14_bn1')
            net = tf.nn.relu(net, name='block14_relu1')
            net = slim.separable_conv2d(net, 2048, [3, 3], scope='block14_dws_conv2')
            net = slim.batch_norm(net, scope='block14_bn2')
            net = tf.nn.relu(net, name='block14_relu2')

            net = slim.avg_pool2d(net, [10, 10], scope='block15_avg_pool')
            # Replace FC layer with conv layer instead
            net = slim.conv2d(net, 2048, [1, 1], scope='block15_conv1')
            logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='block15_conv2')
            logits = tf.squeeze(logits, [1, 2], name='block15_logits')  # Squeeze height and width only
            predictions = slim.softmax(logits, scope='Predictions')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            end_points['Logits'] = logits
            end_points['Predictions'] = predictions

        return logits, end_points


def xception_arg_scope(weight_decay=0.00001,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001):
    '''
    The arg scope for xception model. The weight decay is 1e-5 as seen in the paper.

    INPUTS:
    - weight_decay(float): the weight decay for weights variables in conv2d and separable conv2d
    - batch_norm_decay(float): decay for the moving average of batch_norm momentums.
    - batch_norm_epsilon(float): small float added to variance to avoid dividing by zero.

    OUTPUTS:
    - scope(arg_scope): a tf-slim arg_scope with the parameters needed for xception.
    '''
    # Set weight_decay for weights in conv2d and separable_conv2d layers.
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=None,
                        activation_fn=None):
        # Set parameters for batch_norm. Note: Do not set activation function as it's preset to None already.
        with slim.arg_scope([slim.batch_norm],
                            decay=batch_norm_decay,
                            epsilon=batch_norm_epsilon) as scope:
            return scope
