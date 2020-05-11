import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets


'''

'''






def restore_var(model_path,num_class):

    x = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)

    net,end_points = nets.inception.inception_v2(x,num_classes=num_class)
    output=end_points['Mixed_5c']
    # output=end_points['vgg_16/conv5/conv5_3']
    # train_var=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    train_var=slim.get_trainable_variables()
    # for var in train_var:
    #     print(var)
    #     print(type(var))

    restore_var = slim.get_variables_to_restore(exclude=['InceptionV2/Logits'])
    for var in restore_var:
        print(var)
        print(type(var))


    init = tf.global_variables_initializer()

    variable_restore_op = slim.assign_from_checkpoint_fn(model_path, restore_var,ignore_missing_vars=True)
    data=np.ones(shape=[1,256,256,3],dtype=np.float32)
    with tf.Session() as sess:
        sess.run(init)
        variable_restore_op(sess)

        val=sess.run(output,feed_dict={x:data})
        print(val.shape)

if __name__ == '__main__':
    model_path = '/2_data/share/workspace/xxh/cytological_classification/vgg16_checkpoint/inception_v2.ckpt'
    restore_var(model_path,10)