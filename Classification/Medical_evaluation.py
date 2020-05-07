#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Medical_evaluation.py    
@Contact :   384474737@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
19-10-9 下午3:15   alpha      1.0         None
'''


def caculate_recall(model_1, label):
    tp_1 = 0
    tp_2 = 0
    tp_3 = 0

    precision_0_counts_label = np.sum((label == 0).astype(np.int32))
    precision_2_counts_label = np.sum((label == 2).astype(np.int32))
    precision_3_counts_label = np.sum((label == 3).astype(np.int32))

    precision_0_counts = np.sum((model_1 == 0).astype(np.int32))
    precision_2_counts = np.sum((model_1 == 2).astype(np.int32))
    precision_3_counts = np.sum((model_1 == 3).astype(np.int32))
    # print("precision_x_counts:", precision_0_counts, precision_2_counts, precision_3_counts)
    for index in range(model_1.size):
        if model_1[index] == 0:
            if label[index] == 1 or label[index] == 0:
                tp_1 += 1
        elif model_1[index] == 2 and model_1[index] == label[index]:
            tp_2 += 1
        elif model_1[index] == 3 and model_1[index] == label[index]:
            tp_3 += 1
    # print('tp:', tp_1, tp_2, tp_3)
    recall_1 = tp_1 / (precision_0_counts_label)
    recall_2 = tp_2 / (precision_2_counts_label)
    recall_3 = tp_3 / (precision_3_counts_label)

    print("Sensitivity: o/1+:%.2f, 2+:%.2f ,3+:%.2f" % (recall_1, recall_2, recall_3))


def caculate_specificity(model_1, label):
    tn_1 = 0
    fp_1 = 0

    tn_2 = 0
    fp_2 = 0

    tn_3 = 0
    fp_3 = 0
    for index in range(model_1.size):
        if label[index] != 0 and label[index] == model_1[index]:
            tn_1 += 1
        if label[index] != 0 and model_1[index] == 0:
            fp_1 += 1
    specifity_0 = tn_1 / (tn_1 + fp_1)

    for j in range(model_1.size):
        if label[j] != 2 and label[j] == model_1[j]:
            tn_2 += 1
        if label[j] != 2 and model_1[j] == 2:
            fp_2 += 1
    specifity_2 = tn_2 / (tn_2 + fp_2)

    for k in range(model_1.size):
        if label[k] != 3 and label[k] == model_1[k]:
            tn_3 += 1
        if label[k] != 3 and model_1[k] == 3:
            fp_3 += 1
    specifity_3 = tn_3 / (tn_3 + fp_3)

    print("Specificity: o/1+:%.2f,2+:%.2f,3+:%.2f" % (specifity_0, specifity_2, specifity_3))
