import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import tensorflow as tf

from Classification.xception import xception

NUM_Class = 5

data = np.ones(dtype=np.float32, shape=[1, 299, 299, 3])

x = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32, name="input")
y = tf.placeholder(shape=[None, None, None, NUM_Class], dtype=tf.int32, name="label")

logit = xception.xception(x, NUM_Class)
init = tf.global_variables_initializer()
with tf.Session() as  sess:
    sess.run(init)
    print(logit)
    t = sess.run(logit, feed_dict={x: data})
    print(t.shape)
