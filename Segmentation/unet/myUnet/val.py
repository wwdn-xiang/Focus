import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import myUnet.unet
import logging
import math
from dataset.get_data_from_tfrecords import get_data_from_tfrecords

flags = tf.app.flags
from datetime import datetime

# DATASET INFO
flags.DEFINE_string("dataset_name", 'HE', "the name of dataset")
flags.DEFINE_integer("train_img_count", 15126, "the number of the training dataset")
flags.DEFINE_integer("validation_img_count", 3782, "the number of the validation dataset")
flags.DEFINE_string("dataset_split", 'train', "train/val/test")
flags.DEFINE_string("tfrecords_dir",
                    '/2_data/share/workspace/xxh/HE/HE_Paper/2.5x_5x_tfrecords/',
                    "the folder path of the tfrecords files")

# Trainning param
flags.DEFINE_integer("batch_size", 1, "the batch size when training/val the model.")
flags.DEFINE_integer("epoch", 1, "the batch size when training/val the model.")
flags.DEFINE_float("keep_prob", 1, "the dropout probability.")

# Trainning records
flags.DEFINE_string("checkpoint_dir", '/2_data/share/workspace/xxh/HE/HE_Paper/checkpoint/',
                    'the checkpoint directory of the model saving')
flags.DEFINE_string("log_dir", '/2_data/share/workspace/xxh/HE/HE_Paper/val_log/',
                    'the summaries directory of the model log')

FLAGS = tf.app.flags.FLAGS


def normalize_images(x):
    """normalize input image """
    return tf.subtract(tf.divide(x, 255.0), 0.5) * 2


def de_normalize_image(image):
    out = (image / 2 + 0.5) * 255
    return tf.cast(out, tf.uint8)


def main():
    # reset defual graph
    tf.reset_default_graph()

    # define model
    unet = myUnet.unet.Unet()

    # get data from tfrecords and set batch size and epoch
    images, labels, input_image_image_names = get_data_from_tfrecords(FLAGS.dataset_name, FLAGS.dataset_split,
                                                                      FLAGS.tfrecords_dir, FLAGS.batch_size,
                                                                      FLAGS.epoch)
    # set keep probability
    prediction = unet.build_model(images, FLAGS.keep_prob)

    prediction = tf.argmax(tf.nn.softmax(prediction), axis=3)

    # Add summaries for image.
    for index in range(prediction.shape[0]):
        tf.summary.image("image_%d" % index, tf.expand_dims(de_normalize_image(images[index]), axis=0))
        tf.summary.image("mask_%d" % index, tf.expand_dims(tf.cast(labels[index] * 255, tf.uint8), axis=0))
        tf.summary.image("predictions_%d" % index, tf.expand_dims(tf.expand_dims(tf.cast(prediction[index] * 255, tf.uint8), axis=0),axis=3))
    merge_summary = tf.summary.merge_all()
    # define saver
    saver = tf.train.Saver(max_to_keep=60)

    # setting logging config

    # fetch op
    fetches = {
        'merge_summary': merge_summary,
        'images': images,
        'labels': labels,
        'prediction': prediction,
    }

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())  # global_step
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        for i in range(1):
            result = sess.run(fetches)

            writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)
            writer.add_summary(result['merge_summary'])
        writer.close()
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
