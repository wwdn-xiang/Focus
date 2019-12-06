import os
import random
from glob import glob


def split_train_val_dataset(dataset_dir, ratio, save_txt_dir):
    '''
    Split datasets into train val test datasets.
    :param dataset_dir:
    :param ratio: train_dataset:val_dataset=ratio
    :param save_txt_dir:
    :return:
    '''
    dataset_list = os.listdir(dataset_dir)
    print("Total datasets number is %d" % len(dataset_list))
    train_dataset_conut = int(len(dataset_list) * ratio)

    print('Train dataset number is %d' % train_dataset_conut)
    print('Val dataset number is %d' % (len(dataset_list) - train_dataset_conut))

    train_dataset = random.sample(dataset_list, train_dataset_conut)
    val_dataset = list(set(dataset_list) - set(train_dataset))

    with open(os.path.join(save_txt_dir, "train.txt"), "a+") as tf:
        for image_name in train_dataset:
            tf.writelines(image_name.replace(".jpg", "") + "\n")

    with open(os.path.join(save_txt_dir, "val.txt"), "a+") as tf:
        for image_name in val_dataset:
            tf.writelines(image_name.replace(".jpg", "") + "\n")


if __name__ == '__main__':
    split_train_val_dataset('/2_data/share/workspace/xxh/HE/HE_DATA_135_Finished/2.5x_5x/DATASET/images', 0.8,
                            '/2_data/share/workspace/xxh/HE/HE_Paper/split_dataset_into_train_val')
