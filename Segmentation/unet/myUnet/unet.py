import tensorflow as tf
import tensorflow.contrib.slim as slim


class Unet():
    def __init__(self):
        pass

    def build_model(self, inputs, keep_prob):
        '''
        复现unet模型
        :param inputs:[None,height,width,channel]
        :return:[None,height,width,classes]
        '''
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            padding="SAME",
                            kernel_size=[3, 3],
                            stride=1,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.005)):
            with slim.arg_scope([slim.conv2d_transpose], stride=2):
                with slim.arg_scope([slim.dropout], keep_prob=keep_prob):
                    with tf.name_scope("Unet"):
                        with tf.variable_scope("downsampling"):
                            # downsampling
                            x = slim.conv2d(inputs, 16, scope='conv1')  # 1024
                            x = slim.conv2d(x, 32, scope='conv2')
                            x = slim.conv2d(x, 64, scope='conv3')
                            crop_1 = tf.identity(x, name="crop1")
                            x = slim.dropout(x)

                            x = slim.max_pool2d(x, [2, 2], 2, padding="VALID", scope="max_pool1")  # 512
                            x = slim.conv2d(x, 128, scope='conv4')
                            x = slim.conv2d(x, 128, scope='conv5')
                            crop_2 = tf.identity(x, name="crop2")
                            x = slim.dropout(x)

                            x = slim.max_pool2d(x, [2, 2], 2, padding="VALID", scope="max_pool2")  # 256
                            x = slim.conv2d(x, 256, scope='conv6')
                            x = slim.conv2d(x, 256, scope='conv7')
                            crop_3 = tf.identity(x, name="crop3")
                            x = slim.dropout(x)

                            x = slim.max_pool2d(x, [2, 2], 2, padding="VALID", scope="max_pool3")  # 128
                            x = slim.conv2d(x, 512, scope='conv8')
                            x = slim.conv2d(x, 512, scope='conv9')
                            crop_4 = tf.identity(x, name="crop4")
                            x = slim.dropout(x)

                            x = slim.max_pool2d(x, [2, 2], 2, padding="VALID", scope="max_pool3")  # 64
                            x = slim.conv2d(x, 1024, scope='conv10')
                            x = slim.conv2d(x, 1024, scope='conv11')
                            crop_5 = tf.identity(x, name="crop5")
                            x = slim.dropout(x)

                            x = slim.max_pool2d(x, [2, 2], 2, padding="VALID", scope="max_pool3")  # 32
                            x = slim.conv2d(x, 2048, scope='conv12')
                            x = slim.conv2d(x, 2048, scope='conv13')
                            x = slim.dropout(x)

                        with tf.variable_scope("upsampling"):
                            # upsampling
                            x = slim.conv2d_transpose(x, 1024, scope="deconv1")  # 64
                            x = tf.concat((x, crop_5), axis=3)
                            x = slim.conv2d(x, 1024, scope='upconv1')
                            x = slim.conv2d(x, 1024, scope='upconv2')
                            x = slim.dropout(x)

                            x = slim.conv2d_transpose(x, 512, scope="deconv2")  # 128
                            x = tf.concat((x, crop_4), axis=3)
                            x = slim.conv2d(x, 512, scope='upconv3')
                            x = slim.conv2d(x, 512, scope='upconv4')
                            x = slim.dropout(x)

                            x = slim.conv2d_transpose(x, 256, scope="deconv3")  # 256
                            x = tf.concat((x, crop_3), axis=3)
                            x = slim.conv2d(x, 256, scope='upconv5')
                            x = slim.conv2d(x, 256, scope='upconv6')
                            x = slim.dropout(x)

                            x = slim.conv2d_transpose(x, 128, scope="deconv4")  # 512
                            x = tf.concat((x, crop_2), axis=3)
                            x = slim.conv2d(x, 128, scope='upconv7')
                            x = slim.conv2d(x, 128, scope='upconv8')
                            x = slim.dropout(x)

                            x = slim.conv2d_transpose(x, 64, scope="deconv5")  # 1024
                            x = tf.concat((x, crop_1), axis=3)
                            x = slim.conv2d(x, 64, scope='upconv9')
                            x = slim.conv2d(x, 2, scope='upconv10')

            return x

