#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   uint16_to_uint8.py    
@Contact :   384474737@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
19-7-23 上午9:48   alpha      1.0         None
'''

"""使用skimage模块读取图片，不改变图片数据类型uint16，保存为uint8类型"""

import os

import cv2

import natsort

import numpy as np

from skimage import io

from matplotlib import pylab as plt

input_file = "D:\Deep_Learning\Mouse_data\mouse_difficult\labelme_json\\"  # 文件路径

img_type = ".png"

output_file = "D:\Deep_Learning\Mouse_data\mouse_difficult\cv2_mask\\"

for root, dirs, files in os.walk(input_file, topdown=True):

    for name in natsort.natsorted(dirs):  # natsort,自然排序

        file_name = os.path.join(input_file + name, "label" + img_type)

        midname = name[:name.rindex("_")]

        img = io.imread(file_name)  # Todo:使用skimage模块读取图片不会改变图片的数据格式

        # img = np.array(plt.imread(file_name)) #TODO:不要使用plt.imread读取图片，因为会改变图片的数据格式，uint16读入后会变成float32

        # print(img.shape)

        # io.imshow(img)

        # io.show()

        # plt.imshow(img * 10000) #Todo：img乘以10000是因为uint16位是 0--65531，在MATLAB直接显示时为黑色；标签是1-6

        # plt.axis('off')

        # plt.show()

        print(img.dtype)

        img = img.astype(np.uint8)

        print(img.dtype)

        cv2.imwrite(os.path.join(output_file, midname + img_type), img)  # 路径必须是英文才能正常保存

        # plt.imshow(img * 40) #Todo：img乘以40是因为uint16位是 0--255，在MATLAB直接显示时为黑色 ；标签是1-6

        # plt.show()
