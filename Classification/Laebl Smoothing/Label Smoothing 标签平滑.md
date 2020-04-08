# Label Smoothing 标签平滑

### 1. Label smoothing Regularization(LSR)

​	标签平滑归一化，由名字可以知道，它的优化对象是**Label(Train_y)。**它的优化对象是**Label(Train_y)。**

​	对于分类问题，尤其是多类别分类问题中，常常把类别向量做成[***one-hot*** **vector**](https://blog.csdn.net/ariessurfer/article/details/42526673#0-tsina-1-30478-397232819ff9a47a7b7e80a40613cfe1)**[(独热向量)](https://blog.csdn.net/ariessurfer/article/details/42526673#0-tsina-1-30478-397232819ff9a47a7b7e80a40613cfe1)。**

​	简单地说，就是对于多分类向量，计算机中往往用[0, 1, 3]等此类离散的、随机的而非有序(连续)的向量表示，而one-hot vector 对应的向量便可表示为[0, 1, 0]，即对于长度为n 的数组，只有一个元素是1，其余都为0。因此表征我们已知样本属于某一类别的概率是为1的确定事件，属于其他类别的概率则均为0。

### 2. one-hot 带来的问题

1. ​	无法保证模型的泛化能力，容易造成过拟合；
2. ​    全概率和0概率鼓励所属类别和其他类别之间的差距尽可能加大，而由梯度有界可知，这种情况很难adapt。会造成模型过于相信预测的类别。

### 3. label smoothing 的优化方式

​	对于以Dirac函数分布的真实标签，我们将它变成分为两部分获得（替换）

​	第一部分：将原本Dirac分布的标签变量替换为**(1 - ϵ)的Dirac函数；**

​	第二部分：以概率 ϵ ，在u(k)u(k) 中份分布的随机变量。

```python

def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]    # number of channels or number of classification 类别数
    return ((1-epsilon) * inputs) + (epsilon / K)
```

### 4 . 示例

​	假设我做一个蛋白质二级结构分类，是三分类，那么K=3；

​	假如一个真实标签是[0, 0, 1]，取epsilon = 0.1，

​	新标签就变成了

​	
$$
（1 - 0.1）× [0, 0, 1] + (0.1 / 3) = [0, 0, 0.9] + [0.0333, 0.0333, 0.0333]
$$
​	实际上分了一点概率给其他两类（均匀分），让标签没有那么绝对化，留给学习一点泛化的空间。从而能够提升整体的效果。文章[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567 )
表示，对K = 1000，ϵ = 0.1的优化参数，实验结果有0.2%的性能提升。

