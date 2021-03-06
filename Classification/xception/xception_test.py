#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   xception_test.py
@Contact :   384474737@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
19-7-17 下午5:14   alpha      1.0         None
'''

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf

slim = tf.contrib.slim


class XceptionTest(tf.test.TestCase):

    def testBuild(self):
        batch_size = 5

        height, width = 299, 299

        num_classes = 1001

        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))

            logits, end_points = Classification.xception.xception(inputs, num_classes)

            # Entry Flow

            self.assertEquals(end_points['Xception/block1_res_conv'].get_shape().as_list(), [5, 74, 74, 128])

            self.assertEquals(end_points['Xception/block2_res_conv'].get_shape().as_list(), [5, 37, 37, 256])

            self.assertEquals(end_points['Xception/block3_res_conv'].get_shape().as_list(), [5, 19, 19, 728])

            # Mid Flow

            self.assertEquals(end_points['Xception/block5_dws_conv3'].get_shape().as_list(), [5, 19, 19, 728])

            self.assertEquals(end_points['Xception/block6_dws_conv3'].get_shape().as_list(), [5, 19, 19, 728])

            self.assertEquals(end_points['Xception/block7_dws_conv3'].get_shape().as_list(), [5, 19, 19, 728])

            self.assertEquals(end_points['Xception/block8_dws_conv3'].get_shape().as_list(), [5, 19, 19, 728])

            self.assertEquals(end_points['Xception/block9_dws_conv3'].get_shape().as_list(), [5, 19, 19, 728])

            self.assertEquals(end_points['Xception/block10_dws_conv3'].get_shape().as_list(), [5, 19, 19, 728])

            self.assertEquals(end_points['Xception/block11_dws_conv3'].get_shape().as_list(), [5, 19, 19, 728])

            self.assertEquals(end_points['Xception/block12_dws_conv3'].get_shape().as_list(), [5, 19, 19, 728])

            self.assertEquals(end_points['Xception/block12_res_conv'].get_shape().as_list(), [5, 10, 10, 1024])

            # Exit Flow

            self.assertEquals(end_points['Xception/block14_dws_conv1'].get_shape().as_list(), [5, 10, 10, 1536])

            self.assertEquals(end_points['Xception/block14_dws_conv2'].get_shape().as_list(), [5, 10, 10, 2048])

            self.assertEquals(end_points['Xception/block15_avg_pool'].get_shape().as_list(), [5, 1, 1, 2048])

            self.assertEquals(end_points['Xception/block15_conv1'].get_shape().as_list(), [5, 1, 1, 2048])

            self.assertEquals(end_points['Xception/block15_conv2'].get_shape().as_list(), [5, 1, 1, 1001])

            # Check outputs

            self.assertListEqual(logits.get_shape().as_list(), [batch_size, num_classes])

            self.assertListEqual(end_points['Predictions'].get_shape().as_list(), [batch_size, num_classes])

    def testEvaluation(self):
        batch_size = 1

        height, width = 299, 299

        num_classes = 1001

        with self.test_session():
            eval_inputs = tf.random_uniform((batch_size, height, width, 3))

            logits, _ = Classification.xception.xception(eval_inputs, is_training=False)

            self.assertListEqual(logits.get_shape().as_list(),

                                 [batch_size, num_classes])

            predictions = tf.argmax(logits, 1)

            self.assertListEqual(predictions.get_shape().as_list(), [batch_size])

    def testForward(self):
        batch_size = 1

        height, width = 299, 299

        with self.test_session() as sess:
            inputs = tf.random_uniform((batch_size, height, width, 3))

            logits, _ = Classification.xception.xception(inputs)

            sess.run(tf.global_variables_initializer())

            output = sess.run(logits)

            self.assertTrue(output.any())


if __name__ == '__main__':
    tf.test.main()
