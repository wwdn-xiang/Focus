import os
import random
import shutil
from glob import glob

from tqdm import tqdm

'''




This python script is used to split image dataset into training set and validation set.
The main functions include:
    1. dataset statisitcs
    2. training set and validation set copy or move to the specified training and validation folder.
    3. according to the partition ratio of the dataset,the generation of training and validation text file.
    

Image dataset folder structure.

Classification Dataset_Name:
    Category1:
        Category1_01.jpg
        Category1_02.jpg
        Category1_03.jpg
        ......
    Category2:
        Category2_01.jpg
        Category2_02.jpg
        Category2_03.jpg
        ......
    Category3:
        Category3_01.jpg
        Category3_02.jpg
        Category3_03.jpg
        ......  
    ......
    
Segmentation Dataset_Name
    Image:
        image01.jpg
        image02.jpg
        image03.jpg
        ......
    Mask:
        image01.png
        image02.png
        image03.png

'''


class ImageDataset:
    def __init__(self, image_folder, suffix="jpg"):
        self.image_folder = image_folder
        self.folder_name_list = os.listdir(image_folder)
        self.suffix = suffix

    def dataset_info(self):
        for folder_name in self.folder_name_list:
            folder_abs_path = os.path.join(self.image_folder, folder_name)
            image_list = glob(os.path.join(folder_abs_path, "*.%s" % self.suffix))
            print("{} :{} ".format(folder_name, len(image_list)))

    def train_val_split(self, save_folder, isMove=True, ratio=0.3):
        dataset.dataset_info()
        total_train_count = 0
        total_val_count = 0

        for folder_name in self.folder_name_list:
            train_save_folder = os.path.join(save_folder, "train/%s/" % folder_name)
            val_save_folder = os.path.join(save_folder, "validation/%s/" % folder_name)

            if not os.path.exists(train_save_folder):
                os.makedirs(train_save_folder)
            if not os.path.exists(val_save_folder):
                os.makedirs(val_save_folder)

            folder_abs_path = os.path.join(self.image_folder, folder_name)
            image_list = glob(os.path.join(folder_abs_path, "*.%s" % self.suffix))
            image_count = len(image_list)
            val_count = int(image_count * ratio)

            print("{}---->train:{}---->val:{}".format(folder_name, (image_count - val_count), val_count))

            val_list = random.sample(image_list, val_count)
            train_list = list(set(image_list) ^ set(val_list))

            for train_abs_image in tqdm(train_list):
                image_name = os.path.split(train_abs_image)[-1]
                if isMove:
                    shutil.move(train_abs_image, os.path.join(train_save_folder, image_name))
                else:
                    shutil.copy(train_abs_image, os.path.join(train_save_folder, image_name))
            for val_abs_image in tqdm(val_list):
                image_name = os.path.split(val_abs_image)[-1]
                if isMove:
                    shutil.move(val_abs_image, os.path.join(val_save_folder, image_name))
                else:
                    shutil.copy(val_abs_image, os.path.join(val_save_folder, image_name))
            total_train_count += len(train_list)
            total_val_count += val_count
        print("Total trainning image count :%d" % total_train_count)
        print("Total validation image count :%d" % total_val_count)

    def train_val_split_txt(self, txt_folder, ratio=0.2):
        dataset.dataset_info()
        total_train_count = 0
        total_val_count = 0

        train_txt = os.path.join(txt_folder, "train.txt")
        val_txt = os.path.join(txt_folder, "val.txt")

        if not os.path.exists(train_txt):
            os.mknod(train_txt)
        if not os.path.join(val_txt):
            os.mknod(val_txt)

        with open(train_txt, 'a+') as train_file:
            with open(val_txt, 'a+') as val_file:
                for folder_name in self.folder_name_list:
                    folder_abs_path = os.path.join(self.image_folder, folder_name)
                    image_list = glob(os.path.join(folder_abs_path, "*.%s" % self.suffix))
                    image_count = len(image_list)
                    val_count = int(image_count * ratio)
                    val_list = random.sample(image_list, val_count)
                    train_list = list(set(image_list) ^ set(val_list))

                    for train_abs_image in train_list:
                        image_name = os.path.split(train_abs_image)[-1]
                        train_file.write(image_name + "\n")
                    for val_abs_image in val_list:
                        image_name = os.path.split(val_abs_image)[-1]
                        val_file.write(image_name + "\n")
                    total_train_count += len(train_list)
                    total_val_count += val_count
        print("Total trainning image count :%d" % total_train_count)
        print("Total validation image count :%d" % total_val_count)


class segmentationDataset():
    def __init__(self, image_folder, img_suffix="jpg", mask_suffix='png'):
        self.image_folder = image_folder
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix

        self.image_abs_folder = os.path.join(image_folder, "image")
        self.mask_abs_folder = os.path.join(image_folder, "mask")

    def dataset_info(self):
        image_list = glob(self.image_abs_folder, "*.%s" % self.img_suffix)
        mask_list = glob(self.mask_abs_folder, "*.%s" % self.mask_suffix)
        print("Total image is %d" % (len(image_list)))
        print("Total mask is %d" % (len(mask_list)))
        assert len(image_list) == len(mask_list), "The number of images does not match the mask."

    def train_val_split_txt(self, txt_folder,ratio):
        dataset.dataset_info()
        total_train_count = 0
        total_val_count = 0

        train_txt = os.path.join(txt_folder, "train.txt")
        val_txt = os.path.join(txt_folder, "val.txt")

        if not os.path.exists(train_txt):
            os.mknod(train_txt)
        if not os.path.join(val_txt):
            os.mknod(val_txt)

        with open(train_txt, 'a+') as train_file:
            with open(val_txt, 'a+') as val_file:
                    image_list = glob(os.path.join( self.image_abs_folder, "*.%s" % self.img_suffix))
                    image_count = len(image_list)
                    val_count = int(image_count * ratio)
                    val_list = random.sample(image_list, val_count)
                    train_list = list(set(image_list) ^ set(val_list))

                    for train_abs_image in train_list:
                        image_name = os.path.split(train_abs_image)[-1]
                        train_file.write(image_name[:-4] + "\n")
                    for val_abs_image in val_list:
                        image_name = os.path.split(val_abs_image)[-1]
                        val_file.write(image_name[:-4] + "\n")
                    total_train_count += len(train_list)
                    total_val_count += val_count
        print("Total trainning image count :%d" % total_train_count)
        print("Total validation image count :%d" % total_val_count)


if __name__ == '__main__':
    dataset = ImageDataset('/2_data/share/workspace/xxh/LinesDetection/kankan/areca-nut/')
    dataset.dataset_info()
    dataset.train_val_split_txt("/2_data/share/workspace/xxh/LinesDetection/kankan/image_procession")
