import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

tf.app.flags.DEFINE_string("model_file", './MODEL.pb',
                           "The tensorflow model.pb file.")
tf.app.flags.DEFINE_string("input_name", 'input_1:0',
                           "The op name of the model input. Multiple inputs are separated by commas.")
tf.app.flags.DEFINE_string("output_name", 'output_1:0',
                           "The op name of the model output. Multiple inputs are separated by commas.")

tf.app.flags.DEFINE_enum("precision_mode", 'FP32', ["FP32", 'FP16', 'INT8'], "")
tf.app.flags.DEFINE_integer("batch_size", 1, "The batch size when the inference of the model. ")
tf.app.flags.DEFINE_integer("workspace_size", 1, "The batch size when the inference of the model. ")
FLAGS = tf.app.flags.FLAGS


class CovertTensorRT:
    def __init__(self):
        self.sess = tf.Session()

    def get_tf_graph(self, model_file):
        with gfile.FastGFile(model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            print("Load %s successfully !" % FLAGS.model_file)
        return graph_def

    def generateTRT(self, graph_def):
        trt_graph = trt.create_inference_graph(input_graph_def=graph_def, outputs=[FLAGS.output_name],
                                               max_batch_size=FLAGS.batch_size,
                                               max_workspace_size_bytes=FLAGS.workspace_size,
                                               precision_mode=FLAGS.precision_mode)
        trt_engine_opts = len([1 for n in trt_graph.node if str(n.op) == "TRTEngineOp"])

        print("Successfully convert %d Op to TRTEngineOp." % trt_engine_opts)

        head, tail = os.path.split(FLAGS.model_file)

        name, suffix = tail.split(".")
        trt_model_name = "{0}_trt.{1}".format(name, suffix)

        trt_model_path = os.path.join(head, trt_model_name)

        with gfile.FastGFile(trt_model_path, 'wb')as f:
            f.write(trt_graph.SerializeToString())
        print("save{0}".format(trt_model_path))
        return trt_graph

    def inference(self, graph_def, feed_data):
        tf.import_graph_def(graph_def, name="")
        result = self.sess.run([FLAGS.output_name], feed_dict={FLAGS.input_name: feed_data})
        return result


def run():
    teat_data = np.ones(shape=[1, 16, 16, 3])
    ct = CovertTensorRT()
    graph_def = ct.get_tf_graph(FLAGS.model_file)
    trt_graph_def = ct.generateTRT(graph_def)
    output = ct.inference(trt_graph_def, teat_data)

    print(output)


if __name__ == '__main__':
    run()
