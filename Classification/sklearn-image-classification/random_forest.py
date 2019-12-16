import sklearn
import numpy as np
from dataset_preprocession.data_normarlization import normalization
from sklearn.metrics import accuracy_score, classification_report
from dataset.parse_mnist import get_train_img_label, get_test_img_label

# K-Nearest Neighbor Classification
from sklearn.ensemble import RandomForestClassifier

# 新建随机森林分类器
rfc = RandomForestClassifier()
# 导入训练数据和标签
train_image, train_label = get_train_img_label()
# 图像数据归一化
train_image = normalization(train_image)
# 导入测试数据和标签
test_image, test_label = get_test_img_label()
# 图像数据归一化
test_image = normalization(test_image)
# 模型的训练
rfc.fit(train_image[0:10000,:], train_label[0:10000,:].ravel())
# 测试图像的预测
predict = rfc.predict(test_image[0:400,:])

print("Accuracy score:%.4f" % accuracy_score(predict, test_label[0:400,:]))
print("Classification report for classifier score:%s" % classification_report(predict, test_label[0:400,:]))
