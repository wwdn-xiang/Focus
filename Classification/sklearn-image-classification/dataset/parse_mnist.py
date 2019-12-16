import os
import struct
import numpy as np
from array import array as pyarray


# 加载MNIST数据
def load_mnist(image_file, label_file, path='.'):
    # 生成标签数据[0 1 2 3 4 5 6 7 8 9]
    digits = np.arange(10)
    fname_image = os.path.join(path, image_file)
    fname_label = os.path.join(path, label_file)

    # 标签数据的读取
    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    # 图像数据的读取
    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = np.zeros((N, rows * cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((1, rows * cols))
        labels[i] = lbl[ind[i]]

    return images, labels

    # 获取训练数据及训练标签


def get_train_img_label():
    train_image, train_label = load_mnist("./dataset/train-images.idx3-ubyte", "./dataset/train-labels.idx1-ubyte")
    return train_image, train_label


def get_test_img_label():
    test_image, test_label = load_mnist("./dataset/t10k-images.idx3-ubyte", "./dataset/t10k-labels.idx1-ubyte")
    return test_image, test_label


if __name__ == "__main__":
    pass
    # train_image, train_label = get_train_img_label()
    # print(train_image.shape)
    # print(train_label[0])
