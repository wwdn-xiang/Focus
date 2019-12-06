import os

import tensorflow as tf
from dataset import segmentation_dataset

slim = tf.contrib.slim
dataset_data_provider = slim.dataset_data_provider

_NUM_READERS = 8
_NUM_PREPROCESSING_THREADS = 8


def normalize_images(x):
    """normalize input image """
    if x.dtype != tf.float32:
        x = tf.cast(x, tf.float32)

    return tf.subtract(tf.divide(x, 255.0), 0.5) * 2


def get_data_from_tfrecords(dataset_name, dataset_split, tfrecords_dir, batch_size, epoch):
    dataset = segmentation_dataset.get_dataset(dataset_name, dataset_split,
                                               tfrecords_dir)
    if dataset_split == 'train':
        is_training = True
    else:
        is_training = False

    data_provider = dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=_NUM_READERS,
        num_epochs=epoch if is_training else 1,
        shuffle=is_training)
    [image, image_mask, image_name] = data_provider.get(['image', 'labels_class', 'image_name'])

    image.set_shape([1024, 1024, 3])
    image_mask.set_shape([1024, 1024, 1])

    input_images, input_image_masks, input_image_image_names = tf.train.batch([image, image_mask, image_name],
                                                                              batch_size=batch_size,
                                                                              num_threads=_NUM_PREPROCESSING_THREADS,
                                                                              capacity=200)
    batch_queue = slim.prefetch_queue.prefetch_queue([input_images, input_image_masks, input_image_image_names],
                                                     capacity=20)
    input_images, input_image_masks, input_image_image_names = batch_queue.dequeue()

    input_images = normalize_images(input_images)

    return input_images, input_image_masks, input_image_image_names


'''
if __name__ == '__main__':
    batch_Inputs, batch_Mask = get_data_from_tfrecords("HE", "train",
                                                       "/2_data/share/workspace/xxh/HE/HE_Paper/2.5x_5x_tfrecords", 2)
    print(batch_Inputs.shape)
    print(batch_Inputs)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(init)
        print(sess.run(batch_Inputs)[0])

        coord.request_stop()
        coord.join(threads)
'''
