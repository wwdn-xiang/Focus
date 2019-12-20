import base64
import json
import os
import os.path as osp
import warnings

import PIL.Image
import yaml  # 安装 pip3 install pyyaml

from labelme import utils
import glob

'''
labeme标注的多边形，生成json文件的解析。

注意：生成的.PNG文件 imageio 与PIL.Image读图的差异。
		A.在解析labelme标注生成的json文件时，对于解析的代码中，保存的数据lbl--》xxx.png文件，在读取保存的文件时
		imageio.imread("./xxx.png")文件读取的是三通道数据，max=255 shape=[1024,1024,4]
		np.asarray(Image.open("./xxx.png"))文件读取的是单通道的数据 max=1,shape=[1024,1024]

'''


def json2label(json_folder):
    '''
    遍历文件夹中的所有.json文件
    :param json_folder:
    :return:
    '''
    json_list = glob.glob(os.path.join(json_folder, '*.json'))

    annotations_dir = os.path.join(json_folder, "annotations")
    images_dir = os.path.join(json_folder, "images")
    josn2labels_dir = os.path.join(json_folder, "josn2labels")

    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(josn2labels_dir):
        os.makedirs(josn2labels_dir)

    for json_file in json_list:
        temp_name = json_file.split(".")[0].split("/")[-1]

        out_dir = os.path.join(josn2labels_dir, temp_name)
        if not osp.exists(out_dir):
            os.mkdir(out_dir)

        data = json.load(open(json_file))

        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {'_background_': 0, '1': 1, '2': 2, '3': 3}

        lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name
        lbl_viz = utils.draw_label(lbl, img, label_names)

        PIL.Image.fromarray(img).save(osp.join(out_dir, '%s_img.png' % (temp_name)))
        # 转换为
        utils.lblsave(osp.join(out_dir, '%s_img.png' % (temp_name)), lbl)
        PIL.Image.fromarray(img).save(os.path.join(images_dir, "%s.jpg" % temp_name))
        utils.lblsave(osp.join(annotations_dir, '%s.png' % (temp_name)), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '%s_label_viz.png' % (temp_name)))

        with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            for lbl_name in label_names:
                f.write(lbl_name + '\n')

        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=label_names)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)

        print('Saved to: %s' % out_dir)


def walkFile(file):
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            print(os.path.join(root, f))

        # 遍历所有的文件夹
        for d in dirs:
            print(os.path.join(root, d))


def json2label_walks(json_folder):
    '''
    遍历当前文件夹下所有的子文件夹中的所有.Json文件。
    :param json_folder:
    :return:
    '''
    json_list = []
    for root, dirs, files in os.walk(json_folder):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        # 遍历文件
        for f in files:
            if f.endswith(".json"):
                json_list.append(os.path.join(root, f))

    annotations_dir = os.path.join(json_folder, "annotations")
    images_dir = os.path.join(json_folder, "images")
    josn2labels_dir = os.path.join(json_folder, "josn2labels")

    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(josn2labels_dir):
        os.makedirs(josn2labels_dir)

    for json_file in json_list:
        temp_name = json_file.split(".")[0].split("/")[-1]

        out_dir = os.path.join(josn2labels_dir, temp_name)
        if not osp.exists(out_dir):
            os.mkdir(out_dir)

        data = json.load(open(json_file))

        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {'_background_': 0, '1': 1, '2': 2, '3': 3}

        lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name
        lbl_viz = utils.draw_label(lbl, img, label_names)

        PIL.Image.fromarray(img).save(osp.join(out_dir, '%s_img.png' % (temp_name)))
        # 转换为
        utils.lblsave(osp.join(out_dir, '%s_img.png' % (temp_name)), lbl)
        PIL.Image.fromarray(img).save(os.path.join(images_dir, "%s.jpg" % temp_name))
        utils.lblsave(osp.join(annotations_dir, '%s.png' % (temp_name)), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '%s_label_viz.png' % (temp_name)))

        with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            for lbl_name in label_names:
                f.write(lbl_name + '\n')

        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=label_names)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)

        print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    json_folder = '/home/alpha/Desktop/HE_TMAP_BADCASE_TRAIN/HE_TMAP_Badcase'
    json2label_walks(json_folder)
