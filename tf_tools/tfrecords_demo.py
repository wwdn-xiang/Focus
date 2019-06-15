#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tfrecords_demo.py
@Contact :   384474737@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
19-6-15 下午9:22   alpha      1.0         None
'''
import tensorflow as tf
import numpy as np
import os


# =============================================================================#
# write images and label in tfrecord file and read them out
def encode_to_tfrecords(tfrecords_filename, data_num):
    # write into tfrecord file
    if os.path.exists(tfrecords_filename):
        os.remove(tfrecords_filename)
    writer = tf.python_io.TFRecordWriter('./' + tfrecords_filename)  # 创建.tfrecord文件，准备写入
    for i in range(data_num):
        img_raw = np.random.randint(0, 255, size=(56, 56))
        img_raw = img_raw.tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())
    writer.close()
    return 0


def decode_from_tfrecords(filename_queue, is_batch):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.int64)
    image = tf.reshape(image, [56, 56])
    label = tf.cast(features['label'], tf.int64)

    if is_batch:
        batch_size = 3
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * batch_size
        image, label = tf.train.shuffle_batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=3,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    return image, label


# =============================================================================#

if __name__ == '__main__':
    # make train.tfrecord
    train_filename = "train.tfrecords"
    encode_to_tfrecords(train_filename, 100)
    ##    # make test.tfrecord
    test_filename = 'test.tfrecords'
    encode_to_tfrecords(test_filename, 10)

    #    run_test = True
    filename_queue = tf.train.string_input_producer([train_filename], num_epochs=None)  # 读入流中
    train_image, train_label = decode_from_tfrecords(filename_queue, is_batch=True)

    filename_queue = tf.train.string_input_producer([test_filename], num_epochs=None)  # 读入流中
    test_image, test_label = decode_from_tfrecords(filename_queue, is_batch=True)

    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            # while not coord.should_stop():
            for i in range(2):
                example, l = sess.run([train_image, train_label])  # 在会话中取出image和label
                print('train:')
                print(example, l)
                texample, tl = sess.run([test_image, test_label])
                print('test:')
                print(texample, tl)
        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)

# 另一段tfrecord代码，这段代码实现了float，int和string三种类型数据tfrecord格式的编码和解码。特别注意的是，这里编码和解码时，指定了数据的维度

import tensorflow as tf
import numpy as np

writer = tf.python_io.TFRecordWriter('test.tfrecord')

for i in range(0, 2):
    a = np.random.random(size=(180)).astype(np.float32)
    a = a.data.tolist()
    b = [2016 + i, 2017 + i]
    c = np.array([[0, 1, 2], [3, 4, 5]]) + i
    c = c.astype(np.uint8)
    c_raw = c.tostring()  # 这里是把ｃ换了一种格式存储
    print('  i:', i)
    print('  a:', a)
    print('  b:', b)
    print('  c:', c)
    example = tf.train.Example(features=tf.train.Features(
        feature={'a': tf.train.Feature(float_list=tf.train.FloatList(value=a)),
                 'b': tf.train.Feature(int64_list=tf.train.Int64List(value=b)),
                 'c': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c_raw]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
    print('   writer', i, 'DOWN!')
writer.close()

filename_queue = tf.train.string_input_producer(['test.tfrecord'], num_epochs=None)
# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# get feature from serialized example

features = tf.parse_single_example(serialized_example,
                                   features={
                                       'a': tf.FixedLenFeature([180], tf.float32),
                                       'b': tf.FixedLenFeature([2], tf.int64),
                                       'c': tf.FixedLenFeature([], tf.string)
                                   }
                                   )
a_out = features['a']
b_out = features['b']
c_out = features['c']

print(a_out)
print(b_out)
print(c_out)

a_batch, b_batch, c_batch = tf.train.shuffle_batch([a_out, b_out, c_out], batch_size=3,
                                                   capacity=200, min_after_dequeue=100, num_threads=2)

print(a_batch)
print(b_batch)
print(c_batch)
