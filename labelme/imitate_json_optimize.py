"""半自动标注图像，并生成可供labelme接口解析的json类型的文件"""
import os
import cv2
import glob
import json
from pylab import *
import scipy.io as sio
from json import dumps
import customserializer
from img2json import img_to_json
from base64 import b64encode


def get_mask_contours(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    eroded = cv2.erode(binary, erode_kernel)
    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    imgname, prefix = image_path.split(".")
    save_path = imgname + "_visal." + prefix
    # cv2.imwrite(save_path, img)
    return contours


def trim_contour(contour_list, neighbor=15):
    '''
    :param contour_list: mask's contour list e.g contour_list=[[0,0],[0,1],[1,2]......]
    :param neighbor:neighbor is the The absolute value of the difference between the x and y coordinates of two adjacent points.
    If it's less than neighbor, delete it.The larger neighbor, the more points you have, and vice versa.
    :return:  the trim contour_list
    '''
    import copy
    global temp
    temp = copy.deepcopy(contour_list)
    print("Raw contour len is :%d" % len(contour_list))
    if len(contour_list) > 2:
        for index, piece in enumerate(contour_list):
            if index > 0:
                pre_x, pre_y = contour_list[index - 1]
                x, y = piece
                if abs(x - pre_x) <= neighbor and abs(y - pre_y) <= neighbor:
                    temp.remove(piece)
    print("Trim contour len is :%d" % len(temp))
    return temp


def dict_other_json(imagePath, imageData, shapes, fillColor=None, lineColor=None):
    """

    :param lineColor: list

    :param fillColor: list

    :param imageData: str

    :param imagePath: str

    :return: dict""

    """

    # return {"shapes": shapes, "lineColor": lineColor, "fillColor": fillColor, "imageData": imageData,

    #         "imagePath": imagePath}

    return {"imagePath": imagePath, "imageData": imageData, "shapes": shapes, "fillColor": fillColor,
            "lineColor": lineColor

            }


def dict_shapes(points, label, fill_color=None, line_color=None):
    return {'points': points, 'label': label, 'fill_color': fill_color, 'line_color': line_color}


def coordiante_xy(coordinate_x, coordinate_y):
    return [float(coordinate_x), float(coordinate_y)]


fillColor = [0, 0, 255, 128]

lineColor = [0, 255, 0, 128]


# 参考labelme的json格式重新生成json文件，使用labelme的接口解析数据

class data_mat_read(object):
    """

    读取MATLAB提取的坐标数据MAT文件，结合图像生成的json数据，

    生成一个模仿labelme软件标记object后生成的.json文件

    """

    def __init__(self, img_type=".jpg",
                 img_json_path="/home/xxh/Desktop/tissue/json/",

                 fusion_path="/home/xxh/Desktop/tissue/fusion/",

                 imgPath="/home/xxh/Desktop/tissue/images/",

                 maskPath='/home/xxh/Desktop/tissue/mask/',

                 img_json_type=".json", isMask=True):
        """

        :param coordinate: MATLAB得到的.mat数据文件，包含了每一帧中不同个体的轮廓坐标

        :param img_type: 图片的格式

        :param img_json_path: img2json将图片转成json文件存放的路径

        :param img_json_type: img2json将图片转成json文件的格式

        :param imgPath: 待标注图像的路径

        """

        """

        使用glob从指定路径中获取所有的img_type的json文件

        """

        self.img_json = glob.glob(img_json_path + "/*" + img_json_type)

        self.img_json_path = img_json_path

        self.img_json_type = img_json_type

        self.fusion_path = fusion_path

        self.img_type = img_type

        self.imagePath = imgPath

        self.maskPath = maskPath
        self.isMask = isMask

    def fusion(self):
        """将坐标数据与对应的图像进行融合，生成可以替代labelme生成的json文件"""

        print('-' * 30)

        print("读取坐标，开始生成json文件")

        print('-' * 30)

        img_json_path = self.img_json_path

        img_path = self.imagePath

        img_type = self.img_type

        fusion_path = self.fusion_path
        label = '1'

        mask_list = glob.glob(os.path.join(self.maskPath, "*.png"))

        for mask_abs_path in mask_list:
            shapes = []
            contours = get_mask_contours(mask_abs_path)
            mask_name = os.path.split(mask_abs_path)[-1].split(".")[0]
            imageData = json.load(open(os.path.join(self.img_json_path, mask_name + self.img_json_type)))

            imagePath = mask_name + img_type
            key_name_number = mask_name

            object_num = len(contours)

            if object_num == 0:
                continue
            else:
                for index in range(object_num):
                    contour_ = np.squeeze(contours[index])
                    contour_list = contour_.tolist()
                    # contour_list = trim_contour(contour_list)
                    contour_list = trim_contour(contour_list)
                    if len(contour_list) <= 2:
                        continue
                    else:
                        shapes.append(dict_shapes(contour_list, label))

                data = dict_other_json(imagePath, imageData, shapes, fillColor, lineColor)
                # 写入json文件

                json_file = fusion_path + str(key_name_number) + self.img_json_type

                json.dump(data, open(json_file, 'w'))


if __name__ == "__main__":
    data_read = data_mat_read()
    data_read.fusion()
