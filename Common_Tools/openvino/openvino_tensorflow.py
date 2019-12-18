from openvino.inference_engine import IECore, IEPlugin, IENetwork
import numpy as np
from PIL import Image

data = np.expand_dims(np.asanyarray(Image.open('/home/hx/PycharmProjects/openvino/tes01.jpg').resize((128, 128))),
                      axis=0)	#读取图像
data = np.transpose(data, (0, 3, 2, 1))	#调整shape为【batch,channel,height,with】

net = IENetwork(
    model='/2_data/share/workspace/xxh/Super_Resolution/HE_SuperResolution_Data/train_preupsample_gen_num_repeat_RRDB1/pb/num_repeat_RRDB_1S.xml',
    weights='/2_data/share/workspace/xxh/Super_Resolution/HE_SuperResolution_Data/train_preupsample_gen_num_repeat_RRDB1/pb/num_repeat_RRDB_1S.bin'
    )

plugin = IEPlugin(device='CPU')

exec_net = plugin.load(network=net)
import time

time_list = []
for i in range(50):
    s = time.time()
    res = exec_net.infer({"input": data})
    e = time.time()
    print(e - s)
    time_list.append(e - s)
print(sum(time_list))
