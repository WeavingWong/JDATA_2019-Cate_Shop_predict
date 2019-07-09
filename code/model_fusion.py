#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/18  10:20
# @Author  : chensw、wangwei
# @File    : model_fusion.py
# @Describe: 标明文件实现的功能
# @Modify  : 修改的地方

import pandas as pd

# Step3:模型投票融合
# 读取5个模型输出的预测概率文件
predict_1_4 = pd.read_csv('../data/output/predict/prediction_1_4.csv')
predict_2_4 = pd.read_csv('../data/output/predict/prediction_2_4.csv')
predict_3_4 = pd.read_csv('../data/output/predict/prediction_3_4.csv')
predict_a_4 = pd.read_csv('../data/output/predict/prediction_a_4.csv')
predict_23_4 = pd.read_csv('../data/output/predict/prediction_23_4.csv')

# 根据阈值划分正负样本
print('Start the model fusion!!!')
predict_1_4['label_1'] = predict_1_4.apply(lambda x: 1 if x.prob >= 0.076 else 0, axis=1)
predict_2_4['label_2'] = predict_2_4.apply(lambda x: 1 if x.prob >= 0.057 else 0, axis=1)
predict_3_4['label_3'] = predict_3_4.apply(lambda x: 1 if x.prob >= 0.076 else 0, axis=1)
predict_a_4['label_4'] = predict_a_4.apply(lambda x: 1 if x.prob >= 0.083 else 0, axis=1)
predict_23_4['label_5'] = predict_23_4.apply(lambda x: 1 if x.prob >= 0.073 else 0, axis=1)

test_set = pd.read_csv('../data/features/features_test/test_index.csv')
predict = pd.merge(test_set, predict_1_4, how='inner', on=['user_id', 'cate', 'shop_id'])
predict = pd.merge(predict, predict_2_4, how='inner', on=['user_id', 'cate', 'shop_id'])
predict = pd.merge(predict, predict_3_4, how='inner', on=['user_id', 'cate', 'shop_id'])
predict = pd.merge(predict, predict_a_4, how='inner', on=['user_id', 'cate', 'shop_id'])
predict = pd.merge(predict, predict_23_4, how='inner', on=['user_id', 'cate', 'shop_id'])
predict['label'] = predict['label_1'] + predict['label_2'] + predict['label_3'] + predict['label_4'] + predict['label_5']
predict = predict[(predict.label >= 3)][['user_id', 'cate', 'shop_id']]
print('The model fusion is OK!!!')
predict.to_csv('../data/submit/predict_fusion.csv', index=False, encoding='utf-8')
print(predict.count())
print('Output prediction results!!!')
