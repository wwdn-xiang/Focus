from myUnet import unet

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow as tf
from tensorflow.python.framework import graph_util

_OUTPUT_NODE_NAME = 'output'
_INPUT_NODE_NAME = 'input'


def freeze_model(ckpt_path, model_path):
    '''
    Freeze model from train checkpoint.
    :param ckpt_path: Checkpont path i.e ./gen-126000
    :param model_path: Where the freeze model save.
    :return: None
    '''
    input_data = tf.placeholder(shape=[None, 1024, 1024, 3], dtype=tf.uint8, name=_INPUT_NODE_NAME)
    input_data = tf.cast(input_data, dtype=tf.float32)
    input_data = tf.subtract(tf.divide(input_data, 255.0), 0.5) * 2
    net = unet.Unet()
    prediction = tf.argmax(net.build_model(input_data, 1), axis=3)
    output = tf.identity(prediction, _OUTPUT_NODE_NAME)
    saver = tf.train.Saver()
    print("freeze model from ckpt :", ckpt_path)
    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=_OUTPUT_NODE_NAME.split(",")  # 如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(model_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("finish frozen model!")


if __name__ == '__main__':
    ckpt_path = '/2_data/share/workspace/xxh/HE/HE_Paper/checkpoint/-30801'
    model_path = './myunet.pb'
    freeze_model(ckpt_path, model_path)
