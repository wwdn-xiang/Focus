import sklearn
import numpy as np
from dataset_preprocession.data_normarlization import normalization
from sklearn.metrics import accuracy_score, classification_report
from dataset.parse_mnist import get_train_img_label, get_test_img_label

# K-Nearest Neighbor Classification
from sklearn.ensemble import GradientBoostingClassifier

# 新建GBDT分类器
gbc = GradientBoostingClassifier()
# 导入训练数据和标签
train_image, train_label = get_train_img_label()
# 图像数据归一化
train_image = normalization(train_image)
# 导入测试数据和标签
test_image, test_label = get_test_img_label()
# 图像数据归一化
test_image = normalization(test_image)
# 模型的训练
gbc.fit(train_image[0:1000, :], train_label[0:1000, :].ravel())
# 测试图像的预测
predict = gbc.predict(test_image[0:40, :])

print("Accuracy score:%.4f" % accuracy_score(predict, test_label[0:40, :]))
print("Classification report for classifier score:%s" % classification_report(predict, test_label[0:40, :]))