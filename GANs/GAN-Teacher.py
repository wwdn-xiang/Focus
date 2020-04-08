import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plot

#TODU
#定义生成器
class GeneratorNet:

    def __init__(self):

        self.w1 = tf.Variable(tf.truncated_normal([128,256],stddev=0.01))   #生成器的权重1[128,256]
        self.b1 = tf.Variable(tf.zeros([256],dtype=tf.float32))  #偏值[256]

        self.w2 = tf.Variable(tf.truncated_normal([256, 28*28], stddev=0.01))#权重2[256, 28*28】
        self.b2 = tf.Variable(tf.zeros([28*28], dtype=tf.float32))#偏值2。[28*28]，必须满足输入条件【28*28】

    def forward(self, x):
        y1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        return tf.nn.sigmoid(tf.matmul(y1, self.w2) + self.b2)#【batch_size,28*28】

    def get_params(self):
        return [self.w1,self.w2,self.b1,self.b2]  #保存生成器当前数据状态。

#定义判别器。
class DiscriminatorNet:

    def __init__(self):

        self.w1 = tf.Variable(tf.truncated_normal([28*28, 256], stddev=0.01))  #判别器隐藏层的权重（最好和生成器隐藏层权重一样。256）
        self.b1 = tf.Variable(tf.zeros([256], dtype=tf.float32))

        self.w2 = tf.Variable(tf.truncated_normal([256, 1], stddev=0.01))
        self.b2 = tf.Variable(tf.zeros([1], dtype=tf.float32))

    def forward(self, x):
        y1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        return tf.matmul(y1, self.w2) + self.b2           #【batch_size,1】

    def get_params(self):
        return [self.w1, self.w2, self.b1, self.b2]#保存判别器当前数据状态。

class Net:

    def __init__(self):
        self.x_gen_0 = tf.placeholder(dtype=tf.float32,shape=[None, 128])    #生成的噪点。
        self.y_gen_to_dis_0 = tf.placeholder(dtype=tf.float32,shape=[None,1])  #gen的输出。（标签为0，假数据）

        self.x_real = tf.placeholder(dtype=tf.float32, shape=[None, 28 * 28])  #真数据的样本占位符。
        self.y_real_to_dis = tf.placeholder(dtype=tf.float32, shape=[None, 1]) #真数据的标签。

        self.x_gen_1 = tf.placeholder(dtype=tf.float32, shape=[None, 128])  #生成噪点
        self.y_gen_to_dis_1 = tf.placeholder(dtype=tf.float32, shape=[None, 1]) #伪造成真实数据（给定标签为1），先把真数据和假数据放入判别器训判别器。再把伪造数据放入，从而达到训练生成器的目的。

        self.generatorNet = GeneratorNet()  #实例化生成器
        self.discriminatorNet = DiscriminatorNet()#实例化判别器

#整个网络的前向计算
    def forward(self):
        self.y_gen_0 = self.generatorNet.forward(self.x_gen_0)  #把标记为0的噪点在生成器中进行一次前向计算得到gen_y0。【batch_size,28*28】
        self.output_gen_to_dis_0 = self.discriminatorNet.forward(self.y_gen_0)  #把生成器中标记为0的噪点进行前向计算后得到的值传入判别器中。
        self.output_real_to_dis = self.discriminatorNet.forward(self.x_real)#把真数据传入判别器中。

        self.y_gen_1 = self.generatorNet.forward(self.x_gen_1)#把标记为1的噪点在生成器中进行一次前向计算得到gen_y1。【batch_size,28*28】
        self.output_gen_to_dis_1 = self.discriminatorNet.forward(self.y_gen_1)#把生成器中标记为1的噪点进行前向计算后得到的值传入判别器中

#整个网络的后向计算：
    def backward(self):
        self.loss_gen_to_dis_0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_gen_to_dis_0,labels=self.y_gen_to_dis_0))
        self.loss_real_to_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_real_to_dis,labels=self.y_real_to_dis))
        self.loss_0 = self.loss_gen_to_dis_0 + self.loss_real_to_dis  #把真数据和假数据的误差相加。

        self.loss_gen_to_dis_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_gen_to_dis_1, labels=self.y_gen_to_dis_1)) #把标记为1的假数据传入判别器。求误差。

        self.opt_dis = tf.train.AdamOptimizer().minimize(self.loss_0,var_list=self.discriminatorNet.get_params()) #最小化误差，传入判别器当前数据状态。
        self.opt_gen = tf.train.AdamOptimizer().minimize(self.loss_gen_to_dis_1,var_list=self.generatorNet.get_params())#最小化误差，传入生成器当前数据状态

if __name__ == '__main__':  #主程序的开始。

    net = Net()  #实例化网络
    net.forward()  #前向运算一次
    net.backward() #后向运算一次

    init = tf.global_variables_initializer()  #初始化全局变量。

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  #导入mnist数据集。

    with tf.Session() as sess:
        sess.run(init)  #运行全局变量

        plot.ion()# 打开画图的交互模式
        for i in range(100000):
            x_gen_0 = np.random.uniform(-1., 1., size=[100, 128])  #随机生成噪点。（标记为0，假数据），形状为[100, 128]
            y_gen_to_dis_0 = np.zeros([100, 1])  #lables标记为0.形状【[100, 1]】。

            x_real,_ = mnist.train.next_batch(100)  #获取数据集中地真实数据，批次100.形状(100, 784)
            y_real_to_dis_0 = np.ones([100, 1])#lables标记为1.形状【[100, 1]】
            # print(x_real.shape)

            _g, _r, _= sess.run([net.loss_gen_to_dis_0,net.loss_real_to_dis,net.opt_dis],feed_dict={net.x_gen_0:x_gen_0,net.x_real:x_real,net.y_gen_to_dis_0:y_gen_to_dis_0,net.y_real_to_dis:y_real_to_dis_0})
            #把标记为0的假数据和真实数据放到图中运行。

            x_gen_1 = np.random.uniform(-1., 1., size=[100, 128]) #随机生成噪点。（标记为1，假数据），形状为[100, 128]
            y_gen_to_dis_1 = np.ones([100, 1])#lables标记为1.形状【[100, 1]】
            sess.run(net.opt_gen,feed_dict={net.x_gen_1: x_gen_1, net.y_gen_to_dis_1: y_gen_to_dis_1}) #把标记为1的假数据运行

            if i%100==0:
                test_x_gen = np.random.uniform(-1., 1., size=[1, 128])  #传入1张测试级
                test_y_gen = sess.run(net.y_gen_1, feed_dict={net.x_gen_1: test_x_gen})
                print(test_y_gen)
                img = np.reshape(test_y_gen[0], [28,28])  #把图片转成28*28的格式。

                print(_g,_r)  ##打印出误差。

                plot.clf()
                plot.imshow(img)


                plot.pause(0.001)  #间隔0.001秒。
