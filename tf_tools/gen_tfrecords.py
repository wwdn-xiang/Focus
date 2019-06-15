'''
TFRecords
TFRecords其实是一种二进制文件，虽然它不如其他格式好理解，但是它能更好的利用内存，更方便复制和移动，并且不需要单独的标签文件（等会儿就知道为什么了）… …总而言之，这样的文件格式好处多多，所以让我们用起来吧。

TFRecords文件包含了tf.train.Example 协议内存块(protocol buffer)(协议内存块包含了字段 Features)。我们可以写一段代码获取你的数据， 将数据填入到Example协议内存块(protocol buffer)，将协议内存块序列化为一个字符串，
并且通过tf.python_io.TFRecordWriter 写入到TFRecords文件。

从TFRecords文件中读取数据， 可以使用tf.TFRecordReader的tf.parse_single_example解析器。这个操作可以将Example协议内存块(protocol buffer)解析为张量。

'''
import os
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()
'''
分类Classification 
分割Segmentation 
检测Detection

文件夹目标结构：MODE=1,2.

MODE=1
	分类分类classification：
	./dataset_name/
					+trainning/
						dog/
						panda/
						bike
						car/
						......
					+validation/
						dog/
						panda/
						bike
						car/
						......
					+test/
						dog/
						panda/
						bike
						car/
						......
						
	分割segmentation
		./dataset_name/
					+trainning/
						images/
							image1.jpg
							image2.jpg
							image3.jpg
							......
						seg_mask/
							image1.png
							image2.png
							image3.png
							......
					+validation/
						images/
							image1.jpg
							image2.jpg
							image3.jpg
							......
						seg_mask/
							image1.png
							image2.png
							image3.png
							......
					+test/
						images/
							image1.jpg
							image2.jpg
							image3.jpg
							......
						seg_mask/
							image1.png
							image2.png
							image3.png
							......
	检测Detection
		./dataset_name/
					+trainning/
						images/
							image1.jpg
							image2.jpg
							image3.jpg
							......
						annotations/
							image1.txt/xml
							image2.txt/xml
							image3.txt/xml
							......
					+validation/
						images/
							image1.jpg
							image2.jpg
							image3.jpg
							......
						annotations/
							image1.txt/xml
							image2.txt/xml
							image3.txt/xml
							......
					+test/
						images/
							image1.jpg
							image2.jpg
							image3.jpg
							......
						annotations/
							image1.txt/xml
							image2.txt/xml
							image3.txt/xml
							......
							
MODE=2
	分类分类classification：
		./dataset_name/
						+/images
								dog/
								panda/
								bike
								car/
								......
						+/annotations
								train.txt
								validation.txt
								test.txt
	分割segmentation
		./dataset_name/
						+/images
							image1.jpg
							image2.jpg
							image3.jpg
							......
						+/seg_mask
							image1.png
							image2.png
							image3.png
							......
						+/datase_split
							train.txt
							validation.txt
							test.txt
	检测Detection
		./dataset_name/
						+/images
							image1.jpg
							image2.jpg
							image3.jpg
							......
						+/annotations	
							image1.txt/xml
							image2.txt/xml
							image3.txt/xml
							......
						+/datase_split
							train.txt
							validation.txt
							test.txt
'''

'''
总结
生成tfrecord文件
定义record reader解析tfrecord文件
构造一个批生成器（batcher）
构建其他的操作
初始化所有的操作
启动QueueRunner
'''
# TODO https://www.cnblogs.com/upright/p/6136265.html


dataset_dir = './dataset_name'
writer = tf.python_io.TFRecordWriter("train.tfrecords")
for index, name in enumerate(classes):
    class_path = cwd + name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.resize((224, 224))
    img_raw = img.tobytes()  # 将图片转化为原生bytes
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    writer.write(example.SerializeToString())  # 序列化为字符串
writer.close()

'''
使用队列读取
一旦生成了TFRecords文件，接下来就可以使用队列（queue）读取数据了。
'''


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label


# 之后我们可以在训练的时候这样使用
img, label = read_and_decode("train.tfrecords")

# 使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(3):
        val, l = sess.run([img_batch, label_batch])
        # 我们也可以根据需要对val， l进行处理
        # l = to_categorical(l, 12)
        print(val.shape, l)
