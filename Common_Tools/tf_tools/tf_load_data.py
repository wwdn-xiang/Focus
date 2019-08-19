import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
from glob import glob
import numpy as np

'''
This script is 
关注两个最重要的基础类：Dataset和Iterator。
Dataset可以看作是相同类型“元素”的有序列表。在实际使用时，单个“元素”可以是向量，也可以是字符串、图片，甚至是tuple或者dict。
数据集对象实例化：
dataset = tf.data.Dataset.from_tensor_slices(数据)
迭代器对象实例化(非Eager模式下)：
iterator = dataset.make_one_shot_iterator() 
one_element = iterator.get_next()


dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(one_element))
        
输出：1.0  2.0  3.0  4.0  5.0

读取结束异常：

如果一个dataset中元素被读取完了，再尝试sess.run(one_element)的话，就会抛出tf.errors.OutOfRangeError异常，这个行为与使用队列方式读取数据的行为是一致的。

在实际程序中，可以在外界捕捉这个异常以判断数据是否读取完，综合以上三点请参考下面的代码：
'''

###########################################################一维数据集示范基本使用########################################################
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")

'''
2、高维数据集使用
tf.data.Dataset.from_tensor_slices真正作用是切分传入Tensor的第一个维度，生成相应的dataset，即第一维表明数据集中数据的数量，之后切分batch等操作都以第一维为基础。
dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2)))
传入的数值是一个矩阵，它的形状为(5, 2)，tf.data.Dataset.from_tensor_slices就会切分它形状上的第一个维度，最后生成的dataset中一个含有5个元素，每个元素的形状是(2, )，即每个元素是矩阵的一行。
'''
###########################################################高维数据集示范基本使用########################################################
dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2)))
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")

'''
3、字典使用
在实际使用中，我们可能还希望Dataset中的每个元素具有更复杂的形式，如每个元素是一个Python中的元组，或是Python中的词典。例如，在图像识别问题中，一个元素可以是{“image”: image_tensor, “label”: label_tensor}的形式，这样处理起来更方便，
注意，image_tensor、label_tensor和上面的高维向量一致，第一维表示数据集中数据的数量。相较之下，字典中每一个key值可以看做数据的一个属性，value则存储了所有数据的该属性值。
'''
###########################################################词典数据集示范基本使用########################################################
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    })
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")

'''
复杂的tuple组合数据
类似的，可以使用组合的特征进行拼接，
'''
###########################################################复杂的tuple组合数据集示范基本使用########################################################
dataset = tf.data.Dataset.from_tensor_slices(
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.random.uniform(size=(5, 2)))
)

iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
'''
数据集处理方法

Dataset支持一类特殊的操作：Transformation。一个Dataset通过Transformation变成一个新的Dataset。通常我们可以通过Transformation完成数据变换，打乱，组成batch，生成epoch等一系列操作。
常用的Transformation有：
    map
    batch
    shuffle
    repeat
    
map
和python中的map类似，map接收一个函数，Dataset中的每个元素都会被当作这个函数的输入，并将函数返回值作为新的Dataset
'''

###########################################################dataset.map########################################################
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.map(lambda x: x + 1)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print((sess.run(one_element)))
    except tf.errors.OutOfRangeError:
        print('end')

'''
batch
batch就是将多个元素组合成batch，如上所说，按照输入元素第一个维度，
'''
###########################################################dataset.batch########################################################
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    })

dataset = dataset.batch(2)  # <-----

iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
'''
shuffle

shuffle的功能为打乱dataset中的元素，它有一个参数buffersize，表示打乱时使用的buffer的大小，建议舍的不要太小，一般是1000：
'''
###########################################################dataset.shuffle########################################################
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    })

dataset = dataset.shuffle(buffer_size=5)  # <-----

iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")

'''
repeat

repeat的功能就是将整个序列重复多次，主要用来处理机器学习中的epoch，假设原先的数据是一个epoch，使用repeat(2)就可以将之变成2个epoch：
注意，如果直接调用repeat()的话，生成的序列就会无限重复下去，没有结束，因此也不会抛出tf.errors.OutOfRangeError异常。
'''
###########################################################dataset.repeat########################################################

dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    })

dataset = dataset.repeat(2)  # <-----

iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")

'''
五、模拟读入磁盘图片与对应label
考虑一个简单，但同时也非常常用的例子：读入磁盘中的图片和图片相应的label，并将其打乱，组成batch_size=32的训练样本，在训练时重复10个epoch

在这个过程中，dataset经历三次转变：

    运行dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))后，dataset的一个元素是(filename, label)。filename是图片的文件名，label是图片对应的标签。
    之后通过map，将filename对应的图片读入，并缩放为28x28的大小。此时dataset中的一个元素是(image_resized, label)

    最后，dataset.shuffle(buffersize=1000).batch(32).repeat(10)的功能是：在每个epoch内将图片打乱组成大小为32的batch，并重复10次。最终，dataset中的一个元素是(image_resized_batch, label_batch)，image_resized_batch的形状为(32, 28, 28, 3)，而label_batch的形状为(32, )，接下来我们就可以用这两个Tensor来建立模型了。

'''

###########################################################dataset. read images labels########################################################
'''
# 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label


# 图片文件的列表
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
# label[i]就是图片filenames[i]的label
labels = tf.constant([0, 37, ...])

# 此时dataset中的一个元素是(filename, label)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# 此时dataset中的一个元素是(image_resized, label)
dataset = dataset.map(_parse_function)

# 此时dataset中的一个元素是(image_resized_batch, label_batch)
dataset = dataset.shuffle(buffersize=1000).batch(32).repeat(10)
'''
'''
更多的Dataset创建方法

除了tf.data.Dataset.from_tensor_slices外，目前Dataset API还提供了另外三种创建Dataset的方式：

    tf.data.TextLineDataset()：这个函数的输入是一个文件的列表，输出是一个dataset。dataset中的每一个元素就对应了文件中的一行。可以使用这个函数来读入CSV文件。
    tf.data.FixedLengthRecordDataset()：这个函数的输入是一个文件的列表和一个record_bytes，之后dataset的每一个元素就是文件中固定字节数record_bytes的内容。通常用来读取以二进制形式保存的文件，如CIFAR10数据集就是这种形式。
    tf.data.TFRecordDataset()：顾名思义，这个函数是用来读TFRecord文件的，dataset中的每一个元素就是一个TFExample。

'''
'''
更多的Iterator创建方法

在非Eager模式下，最简单的创建Iterator的方法就是通过dataset.make_one_shot_iterator()来创建一个one shot iterator。

除了这种one shot iterator外，还有三个更复杂的Iterator，即：

    initializable iterator
    reinitializable iterator
    feedable iterator

initializable iterator方法要在使用前通过sess.run()来初始化，使用initializable iterator，可以将placeholder代入Iterator中，实现更为灵活的数据载入，实际上占位符引入了dataset对象创建中，我们可以通过feed来控制数据集合的实际情况。
'''
limit = tf.placeholder(dtype=tf.int32, shape=[])
dataset = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=limit))

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={limit: 10})
    for i in range(10):
        value = sess.run(next_element)
        print(value)
        assert i == value
'''
initializable iterator还有一个功能：读入较大的数组。

在使用tf.data.Dataset.from_tensor_slices(array)时，实际上发生的事情是将array作为一个tf.constants保存到了计算图中。
当array很大时，会导致计算图变得很大，给传输、保存带来不便。这时，我们可以用一个placeholder取代这里的array，并使用initializable iterator，只在需要
时将array传进去，这样就可以避免把大数组保存在图里，示例代码为（来自官方例程）：

可见，在上面程序中，feed也遵循着类似字典一样的规则，创建两个占位符(keys)，给data_holder去feed数据文件，给label_holder去feed标签文件。

reinitializable iterator和feedable iterator相比initializable iterator更复杂，也更加少用，如果想要了解它们的功能，可以参阅官方介绍，这里就不再赘述了。
'''
# 从硬盘中读入两个Numpy数组
with np.load("/var/data/training_data.npy") as data:
    features = data["features"]
    labels = data["labels"]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
iterator = dataset.make_initializable_iterator()
sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
'''


def _parse_function_(image_name, image_label):
    image_string = tf.read_file(image_name)
    image_data = tf.image.decode_jpeg(image_string)

    label_string = tf.read_file(image_label)
    image_label = tf.image.decode_png(label_string)

    return image_data, image_label, image_name


def get_image_labels(image_abpath_list):
    image_labels = []
    for image_abspath in image_abpath_list:
        str_label = image_abspath.split("/")[-1].split("_")[0]
        if str_label == 'ab':
            image_label = 4
        else:
            image_label = int(str_label[0])

        image_labels.append(image_label)
    return image_labels


def get_data_iterator(image_abpaths_list, image_labels_list, buffer_size, batch_size, epoch):
    dataset = tf.data.Dataset.from_tensor_slices((image_abpaths_list, image_labels_list))
    dataset = dataset.map(_parse_function_)
    dataset = dataset.shuffle(buffer_size=(buffer_size)).batch(batch_size).repeat(epoch)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


image_train_dir = "/1_data/xxh_workspace/SEG_DCIS_IDC/data/x5_data_1024/HE/debug/images/"
images_train_abs_path_list = glob(os.path.join(image_train_dir, "*.jpg"))

anno_train_dir = "/1_data/xxh_workspace/SEG_DCIS_IDC/data/x5_data_1024/HE/debug/annotations/"
anno_train_abs_path_list = glob(os.path.join(anno_train_dir, "*.png"))
iterator = get_data_iterator(images_train_abs_path_list, anno_train_abs_path_list, 15, 2, 1)
with tf.Session() as sess:
    while True:
        image_data, image_label, image_name = sess.run(iterator.get_next())
        print(image_name)
'''
'''
********************************************************************************************
TensorFlow tf.data 导入数据（tf.data官方教程） * * * * *
https://blog.csdn.net/u014061630/article/details/80728694
https://blog.csdn.net/u014061630/article/details/80712635#1_tfdata_API_14
********************************************************************************************

'''
