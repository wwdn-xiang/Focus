#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py    
@Contact :   384474737@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
19-6-13 下午3:30   alpha      1.0         None
'''

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import graph_util
from nets import inception_v3
from nets import nets_factory


class TF_Utils(object):
    def __init__(self):
        pass

    @staticmethod
    def get_op_name_value_from_ckpt(checkpoint_path):
        '''
        :param checkpoint_path: e.g. "./train_logs/model.ckpt-16926"
        :return: None
        '''
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print("tensor_name: ", key)
            print(reader.get_tensor(key))

    @staticmethod
    def get_op_name_value_from_pb(pb_path):
        '''
        :param pb_path: e.g. './freeze_model/fcn_model.pb'
        :return: None
        '''

        with tf.gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        for index in range(len(tensor_name_list)):
            tensor_name = tensor_name_list[index]
            tensor_name = tensor_name + ":0"
            print(tensor_name)

    @staticmethod
    def freeze_graph(input_checkpoint, frozen_model):
        '''
        :param input_checkpoint: e.g. "./train_logs/model.ckpt-58500"
        :param frozen_model:  e.g. './freeze_model/fcn_model.pb'
        :return:
        '''
        output_node_name = 'classes'
        saver = tf.train.import_meta_graph(input_checkpoint + ".meta", clear_devices=True)
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        with tf.Session() as sess:
            saver.restore(sess, input_checkpoint)  # restore data and graph
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=output_node_name.split(",")  # 如果有多个输出节点，以逗号隔开
            )
            with tf.gfile.GFile(frozen_model, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("finish frozen model!")

    @staticmethod
    def pb2tfboard(cls, tf_pb_path):
        '''
        show the model structure graph
        :param tf_pb_path: e.g. './freeze_model/fcn_model.pb'
        :return: None
        '''
        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        graph_def.ParseFromString(tf.gfile.FastGFile(tf_pb_path, 'rb').read())
        tf.import_graph_def(graph_def, name='graph')
        summaryWriter = tf.summary.FileWriter('log/', graph)

    @staticmethod
    def freeze_model_from_forward_ckpt(ckpt_dir, output_node_name, model_path):
        '''
        :param ckpt_dir: the train checkpoint dir e.g. './train_logs/'
        :param output_node_name: e.g. 'InceptionV3/Logits/classes,output'(multi-nodes xx1,xx2,xx3...) or 'output'(only one node)
        :param model_path:  the freeze model save path e.g.'./model.pb'
        :return:
        '''
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        ckpt_path = ckpt.model_checkpoint_path
        network_fn = nets_factory.get_network_fn('inception_v3', 5)
        x = tf.placeholder(shape=[None, None, None, 3], dtype=tf.uint8, name="input")
        logits, end_points = network_fn(x)
        saver = tf.train.Saver()
        print("freeze model from ckpt :", ckpt_path)
        with tf.Session() as sess:
            saver.restore(sess, ckpt_path)
            graph = tf.get_default_graph()
            input_graph_def = graph.as_graph_def()
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=output_node_name.split(",")  # 如果有多个输出节点，以逗号隔开
            )
            with tf.gfile.GFile(model_path, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("finish frozen model!")

    @staticmethod
    def _parse_function_(image_name, image_label):
        image_string = tf.read_file(image_name)
        image_data = tf.image.decode_jpeg(image_string)

        label_string = tf.read_file(image_label)
        image_label = tf.image.decode_png(label_string)

        return image_data, image_label

    @staticmethod
    def get_seg_data_iterator(image_abpaths_list, image_labels_list, buffer_size, batch_size, epoch):

        dataset = tf.data.Dataset.from_tensor_slices((image_abpaths_list, image_labels_list))
        dataset = dataset.map(TF_Utils._parse_function_)
        dataset = dataset.shuffle(buffer_size=(buffer_size)).batch(batch_size).repeat(epoch)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


if __name__ == '__main__':
    TF_Utils.pb2tfboard("/1_data/xxh_workspace/SEG_DCIS_IDC/freeze_model/fcn_model.pb")
