import sklearn
import numpy as np
from dataset_preprocession.data_normarlization import normalization
from sklearn.metrics import accuracy_score, classification_report
from dataset.parse_mnist import get_train_img_label, get_test_img_label

# K-Nearest Neighbor Classification
from sklearn.svm import SVC

# 新建支持向量机分类器
svc = SVC()
# 导入训练数据和标签
train_image, train_label = get_train_img_label()
# 图像数据归一化
train_image = normalization(train_image)
# 导入测试数据和标签
test_image, test_label = get_test_img_label()
# 图像数据归一化
test_image = normalization(test_image)
# 模型的训练
svc.fit(train_image[0:1000, :], train_label[0:1000, :].ravel())
# 测试图像的预测
predict = svc.predict(test_image[0:40, :])

print("Accuracy score:%.4f" % accuracy_score(predict, test_label[0:40, :]))
print("Classification report for classifier score:%s" % classification_report(predict, test_label[0:40, :]))
