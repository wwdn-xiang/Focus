import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import myUnet.unet
import logging
import numpy as np
import math
from sklearn.metrics import classification_report
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
flags.DEFINE_integer("epoch", 5, "the batch size when training/val the model.")
flags.DEFINE_float("keep_prob", 0.75, "the dropout probability.")

# define learning rate and decay
flags.DEFINE_float("learning_rate", 0.01, "the initial learning ratio.")
flags.DEFINE_integer("decay_steps", 1000, "How many steps are iterate attenuate once.")
flags.DEFINE_float("decay_rate", 0.96, "the decay rate.")

# Trainning records
flags.DEFINE_string("checkpoint_dir", '/2_data/share/workspace/xxh/HE/HE_Paper/checkpoint/',
                    'the checkpoint directory of the model saving')
flags.DEFINE_string("log_dir", '/2_data/share/workspace/xxh/HE/HE_Paper/log/',
                    'the summaries directory of the model log')
flags.DEFINE_integer("save_checkpoint_frequency", 100, 'The frequency at which checkpoint is saved')
flags.DEFINE_integer("save_log_frequency", 100, 'The frequency at which log is saved')
flags.DEFINE_integer("show_log_frequency", 10, 'The frequency show the train log')

FLAGS = tf.app.flags.FLAGS


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

    # preprocessing labels
    labels = tf.cast(labels, tf.int32)

    # define loss
    seg_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(labels, axis=3), logits=prediction))
    l2_loss = tf.losses.get_regularization_loss()

    total_loss = seg_loss + l2_loss

    # set global step variable
    global_step = tf.Variable(0, trainable=False)

    # define learning rate declay func
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate,
                                               staircase=True)

    # define optimizer
    optmizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step)

    # prediction_label for metrics
    prediction_label = tf.argmax(prediction, axis=3)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for model variables.
    for model_var in tf.trainable_variables():
        summaries.add(tf.summary.histogram(model_var.op.name, model_var))

    # Add summaries for scalar
    summaries.add(tf.summary.scalar('seg_loss', seg_loss))
    summaries.add(tf.summary.scalar('l2_loss', l2_loss))
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    summaries.add(tf.summary.scalar('global_step', global_step))
    summaries.add(tf.summary.scalar('learning_rate', FLAGS.learning_rate))

    # Merge all the summaries, and get the merge summaries op
    merge_summary = tf.summary.merge_all()

    # define saver
    saver = tf.train.Saver(max_to_keep=60)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    # setting logging config

    logfile = '{0}/training_logfile_{1}.log'.format(FLAGS.log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    formatter = '%(levelname)s:%(asctime)s:%(message)s'
    logging.basicConfig(level=logging.INFO, filename=logfile, format=formatter, datefmt='%Y-%m-%d %I:%M:%S')

    # fetch op
    fetches = {
        'optmizer': optmizer,
        'seg_loss': seg_loss,
        'l2_loss': l2_loss,
        'total_loss': total_loss,
        'prediction_label': prediction_label,
        'labels': labels,
        'global_step': global_step,
        'learning_rate': learning_rate,
        'merge_summary': merge_summary
    }

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())  # global_step
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        total_iter_step = int(FLAGS.epoch * FLAGS.train_img_count / FLAGS.batch_size)

        writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)
        old_step = 0
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
            old_step = int(ckpt.model_checkpoint_path.split("-")[-1])
        for iter_step in range(total_iter_step):
            iter_step += old_step
            epoch_step = math.ceil(iter_step * FLAGS.batch_size / FLAGS.train_img_count)
            if epoch_step <= 0:
                epoch_step = 1
            result = sess.run(fetches=fetches)
            if iter_step % FLAGS.save_checkpoint_frequency == 0:
                saver.save(sess, FLAGS.checkpoint_dir, global_step=result['global_step'])

            if iter_step % FLAGS.save_log_frequency == 0:
                writer.add_summary(result['merge_summary'], global_step=result['global_step'])
            if iter_step % FLAGS.show_log_frequency == 0:

                print(result['labels'].flatten())
                print(result['prediction_label'].flatten())

                report_metrcis = classification_report(result['labels'].flatten(), result['prediction_label'].flatten())

                msg_info = "Training log iter_step-->{:7}   epoch-->{}  total_loss-->{:.6f}  seg_loss-->{:.6f}  l2_loss-->{:.6f}".format(
                    iter_step, epoch_step, result['total_loss'], result['seg_loss'], result['l2_loss'],
                    result['l2_loss'])

                logging.info(msg_info)
                print(msg_info)
                print(report_metrcis)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
