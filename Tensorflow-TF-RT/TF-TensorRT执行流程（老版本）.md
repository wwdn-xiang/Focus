TF-TensorRT执行流程（老版本）

https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#prereqs

TF-RT优化内容:

- 无用的输出层
- 卷积,偏置,RELU激活等层融合成一层(certain layers (such as convolution, bias, and ReLU) are fused to form a single layer)
- Horizontal layer fusion improves performance by combining layers that take the same source tensor and apply the same operations with  similar parameters

在执行阶段,TF-TRT执行兼容的子图(可以优化的部分图中的layer----TensorRT will parse the model and apply optimizations to the portions of the graph wherever possible. ),Tensorflow执行剩余的部分

**Note: These graph optimizations do not change the underlying computation in the graph; instead, they look to restructure the graph to perform the operations much faster and more efficiently. **

**本质:图的优化没有改变运行计算的本质,只是重构了图,使其运行更快,更高效.TF-TRT是Tensorflow的一部分**



支持的op:

https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/uff/Operators.html



自定义op:

https://devtalk.nvidia.com/default/topic/1059051/tensorrt/warning-no-conversion-function-registered-for-layer/



博客示例:

https://www.cnblogs.com/573177885qq/p/11944607.html



模型加速流程:

TensorRT: https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html

![](C:\Users\16326\Desktop\TensorRT加速流程.jpg)