import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
import cv2
import copy


class augument:
    def __init__(self,img_width,img_height):
        self.img_width=img_width
        self.img_height=img_height

    def random_vector(self,min,max):
        """生成范围矩阵"""
        min=np.array(min)
        max=np.array(max)
        assert min.shape==max.shape
        assert len(min.shape) == 1
        return np.random.uniform(min, max)

    def basic_matrix(self,translation):
        """基础变换矩阵"""
        return np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])

    def adjust_matrix_for_image(self,trans_matrix):
        """根据图像调整当前变换矩阵"""
        transform_matrix=copy.deepcopy(trans_matrix)
        transform_matrix[0:2, 2] *= [self.img_width, self.img_height]
        center = np.array((0.5 * self.img_width, 0.5 * self.img_height))
        transform_matrix = np.linalg.multi_dot([self.basic_matrix(center), transform_matrix, self.basic_matrix(-center)])
        return transform_matrix

    def apply_transform_img(self,img,transform):
        """仿射变换"""
        output = cv2.warpAffine(img, transform[:2, :], dsize=(img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderValue=0)   #cv2.BORDER_REPLICATE,cv2.BORDER_TRANSPARENT   borderMode=cv2.BORDER_TRANSPARENT,
        return output

    def apply_transform_boxes(self,boxes,transform):
        """仿射变换"""
        #TODO

    def apply_transform_points(self,points,transform):
        """仿射变换"""
        #TODO

    def get_translation_matrix(self,min_factor=(-0.1, -0.1), max_factor=(0.1, 0.1),adjust=True):
        factor = self.random_vector(min_factor, max_factor)
        matrix = np.array([[1, 0, factor[0]], [0, 1, factor[1]], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    def random_translation_np(self,img,min_factor=(-0.1, -0.1), max_factor=(0.1, 0.1)):
        matrix=self.get_translation_matrix(min_factor=min_factor, max_factor=max_factor)
        out_img=self.apply_transform_img(img, matrix)
        return  out_img

    def random_translation(self,img,min_factor=(-0.1, -0.1), max_factor=(0.1, 0.1)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_translation_np, [img,min_factor ,max_factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def get_flip_matrix(self,adjust=True):
        factors=np.array([[1,1],[1,-1],[-1,1]])
        idx=np.random.randint(3,size=1)
        factor=factors[idx]
        matrix = np.array([[factor[0][0], 0, 0],[0, factor[0][1], 0],[0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    def random_flip_np(self,img):
        matrix=self.get_flip_matrix()
        out_img=self.apply_transform_img(img, matrix)
        return  out_img

    def random_flip(self, img):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_flip_np, [img], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def get_rotate_matrix(self,factor=(-0.2, 0.2),adjust=True):
        angle = np.random.uniform(factor[0], factor[1])
        matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    def random_rotate_np(self,img,factor=(-0.2, 0.2)):
        matrix=self.get_rotate_matrix(factor=factor)
        out_img=self.apply_transform_img(img, matrix)
        return  out_img

    def random_rotate(self, img,factor=(-0.2, 0.2)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_rotate_np, [img,factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def get_scale_matrix(self,min_factor=(0.8, 0.8), max_factor=(1.2, 1.2),adjust=True):
        factor = self.random_vector(min_factor, max_factor)
        matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    def random_scale_np(self,img,min_factor=(0.8, 0.8), max_factor=(1.2, 1.2)):
            matrix=self.get_scale_matrix(min_factor=min_factor, max_factor=max_factor)
            out_img=self.apply_transform_img(img, matrix)
            return  out_img

    def random_scale(self, img,min_factor=(0.9, 0.9), max_factor=(1.2, 1.2)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_scale_np, [img,min_factor,max_factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def get_shear_matrix(self, factor=(-0.4,0.4),adjust=True):
        factor1 = np.random.uniform(factor[0], factor[1])
        factor2 = np.random.uniform(factor[0], factor[1])
        matrix = np.array([[1, factor1, 0], [factor2, 1, 0], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    def random_shear_np(self,img,factor=(-0.2,0.2)):
        matrix = self.get_shear_matrix(factor=factor)
        out_img = self.apply_transform_img(img, matrix)
        return out_img

    def random_shear(self, img,factor=(-0.2,0.2)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_shear_np, [img,factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def get_combination_affine_matrix(self,adjust=True):
        t1 = self.get_translation_matrix(min_factor=(-0.15, -0.15), max_factor=(0.15, 0.15),adjust=False)
        t2 = self.get_flip_matrix(adjust=False)
        t3 = self.get_rotate_matrix(factor=(-0.3, 0.3),adjust=False)
        t4 = self.get_scale_matrix(min_factor=(0.8, 0.8), max_factor=(1.2, 1.2),adjust=False)
        # t5 = self.get_shear_matrix( factor=(-0.2,0.2),adjust=False)
        matrix = np.linalg.multi_dot([t1, t2, t3, t4])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    def random_combination_affine_np(self,img,boxes=None,points=None):
        matrix = self.get_combination_affine_matrix()
        out_img = self.apply_transform_img(img, matrix)
        if boxes!=None:
            out_boxes = self.apply_transform_boxes(boxes, matrix)
        if points!=None:
            out_points = self.apply_transform_points(points, matrix)
        return out_img

    def random_combination_affine(self, img):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_combination_affine_np, [img], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def img_contrast(self,img,coefficient):
        out_img=img*coefficient
        out_img[out_img > 255] = 255
        out_img =out_img.astype(np.uint8)
        return out_img

    def random_contrast_np(self,img,factor=(0.6,1.5)):
        coefficient = np.random.uniform(factor[0], factor[1])
        out_img= self.img_contrast(img,coefficient)
        return out_img

    def random_contrast(self,img,factor=(0.6,1.5)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_contrast_np, [img,factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def img_bright(self, img, coefficient):
        out_img = img + 255*coefficient
        out_img[out_img > 255 ] = 255
        out_img[out_img < 0] = 0
        out_img = out_img.astype(np.uint8)
        return out_img

    def random_bright_np(self, img, factor=(-0.2, 0.2)):
        coefficient = np.random.uniform(factor[0], factor[1])
        out_img = self.img_bright(img, coefficient)
        return out_img

    def random_bright(self, img, factor=(-0.2, 0.2)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_bright_np, [img, factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def img_saturation(self, img, coefficient):
        out_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        out_img[:,:,1]=out_img[:,:,1]*coefficient
        out_img[out_img[:,:,1] > 255 ] = 255
        out_img = out_img.astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2RGB)
        return out_img

    def random_saturation_np(self, img, factor=(0.7, 1.3)):
        coefficient = np.random.uniform(factor[0], factor[1])
        out_img = self.img_saturation(img, coefficient)
        return out_img

    def random_saturation(self, img, factor=(0.7, 1.3)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_saturation_np, [img, factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def img_hue(self, img, coefficient):
        out_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        out_img[:,:,0]=out_img[:,:,0]+45*coefficient
        out_img = out_img.astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2RGB)
        return out_img

    def random_hue_np(self, img, factor=(-0.1, 0.1)):
        coefficient = np.random.uniform(factor[0], factor[1])
        out_img = self.img_hue(img, coefficient)
        return out_img

    def random_hue(self, img, factor=(-0.1, 0.1)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_hue_np, [img, factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def random_combination_color_np(self,img):
        coefficient = np.random.uniform(0.9,1.1)
        out_img = self.img_contrast(img, coefficient)
        coefficient = np.random.uniform(-0.05, 0.05)
        out_img = self.img_bright(out_img, coefficient)
        coefficient = np.random.uniform(0.7, 1.3)
        out_img = self.img_saturation(out_img, coefficient)
        coefficient = np.random.uniform(-0.1, 0.1)
        out_img = self.img_hue(out_img, coefficient)
        return out_img

    def random_combination_color(self, img):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_combination_color_np, [img], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def random_combination(self,img,using_affine=True,using_color=True):
        out_img=copy.copy(img)
        if using_affine:
            out_img=self.random_combination_affine(out_img)
        if using_color:
            out_img = self.random_combination_color(out_img)
        return out_img

class transform(augument):
    def __init__(self,img_width,img_height):
        super(transform,self).__init__(img_width,img_height)

    @staticmethod
    def imread(img_path):
        img = cv2.imread(img_path)
        cv2.cvtColor(img,cv2.COLOR_BGR2RGB,img)
        return img

    @staticmethod
    def imwrite(img_path, img):
        if len(img.shape)==3 and img.shape[2]==3:
            img = copy.deepcopy(img)
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR,img)
        cv2.imwrite(img_path, img)

    @staticmethod
    def imshow(img):
        cv2.imshow("img", img)
        cv2.waitKey(0)



if __name__=='__main__':
    tran = transform(1074,1019)
    input = transform.imread('/2_data/share/workspace/xxh/DCIS_IDC_Classifier/IDC_3_17010003_9333_4131_img.jpg')
    img = tf.convert_to_tensor(input)
    sess = tf.Session()

    for i in range(10):
        # img2 = tran.random_translation(img,(-0.2,-0.2),(0.3,0.3))
        # matrix = tran.get_translation_matrix((-0.1,-0.1),(0.1,0.1))

        # img2 = tran.random_flip(img)
        # matrix = tran.get_flip_matrix()

        # img2 = tran.random_rotate(img,(-0.3,0.3))
        # matrix = tran.get_rotate_matrix((-0.3,0.3))

        # img2 = tran.random_scale(img,(0.8,0.8),(1.2,1.2))
        # matrix = tran.get_scale_matrix((0.8,0.8),(1.2,1.2))

        # img2 = tran.random_shear(img,(-0.2,0.2))
        # matrix = tran.get_shear_matrix((-0.1,0.1))

        # img2 = tran.random_combination_affine(img)
        # matrix = tran.get_combination_affine_matrix()

        # img2 = tran.random_contrast(img)
        # img2 = tran.random_bright(img)
        # img2 = tran.random_saturation(img)
        # img2 = tran.random_hue(img)
        # img2 = tran.random_combination_color(img)

        img2 =tran.random_combination(img)
        img2_ = sess.run(img2)
        transform.imwrite('/home/ljw/data/x'+str(i)+'.png', img2_)
