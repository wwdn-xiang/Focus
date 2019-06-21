#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   aspp.py    
@Contact :   384474737@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
19-6-21 上午8:28   alpha      1.0         None
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim


def AtrousSpatialPyramidPoolingModule(inputs, depth=256):
    feature_map_size = tf.shape(inputs)

    # Global average pooling
    image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    image_features = slim.conv2d(image_features, depth, [1, 1], activation_fn=None)

    # resize_bilinear feature map shape [input_fm_height,input_fm_width]
    image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

    atrous_pool_block_1 = slim.conv2d(inputs, depth, [1, 1], activation_fn=None)

    atrous_pool_block_6 = slim.conv2d(inputs, depth, [3, 3], rate=6, activation_fn=None)

    atrous_pool_block_12 = slim.conv2d(inputs, depth, [3, 3], rate=12, activation_fn=None)

    atrous_pool_block_18 = slim.conv2d(inputs, depth, [3, 3], rate=18, activation_fn=None)

    net = tf.concat(
        (image_features, atrous_pool_block_1, atrous_pool_block_6, atrous_pool_block_12, atrous_pool_block_18), axis=3)
    net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)

    return net
