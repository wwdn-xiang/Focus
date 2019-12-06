import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from glob import glob
import numpy as np


def load_model(model_pa_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = model_pa_path
        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            net_input = sess.graph.get_tensor_by_name("input:0")
            net_output = sess.graph.get_tensor_by_name("output:0")

            raw_img = Image.open('./17010002 x5_24576_57344.jpg')
            value = np.expand_dims(np.asarray(raw_img), axis=0)
            seg_output = sess.run(net_output, feed_dict={net_input: value})
            seg_output[seg_output == 1] = 255
            mask_img = Image.fromarray(np.squeeze(seg_output).astype(np.uint8))
            mask_img.save("pre.jpg")


if __name__ == '__main__':
    load_model('./myunet.pb')
