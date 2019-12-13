import os
import numpy as np
from PIL import Image

'''
将labelme解析的.png文件，生成为单通道的，可视化全黑的.png文件。因为tensorflow读取的结果与PIL.Image读取的结果不一致。
例如： PIL.Image读取像素max_value=1 ，tf.image.decode_jpg/png解析的结果为max_value=38。
在生成TFRecords时，需要进行转换。
'''


def convert_mask_for_train(mask_dir, save_dir):
    mask_images = os.listdir(mask_dir)
    for index, mask_img_name in enumerate(mask_images):
        raw_data = Image.open(os.path.join(mask_dir, mask_img_name))
        np_data = np.asarray(raw_data)
        covert_data = Image.fromarray(np_data, mode='L')
        covert_data.save(os.path.join(save_dir, mask_img_name))


if __name__ == '__main__':
    convert_mask_for_train('/2_data/share/workspace/xxh/HE/HE_DATA_135_Finished/2.5x_5x/DATASET/HE_TMAP/annotations/',
                           '/2_data/share/workspace/xxh/HE/HE_DATA_135_Finished/2.5x_5x/DATASET/HE_TMAP/mask_for_train/')
