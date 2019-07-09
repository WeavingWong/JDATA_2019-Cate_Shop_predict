#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/18  10:20
# @Author  : chensw、wangwei
# @File    : test.py
# @Describe: 标明文件实现的功能
# @Modify  : 修改的地方

import pandas as pd


def get_train_set_label(train_start_time, train_end_time, label_start_time, label_end_time, path):
    # 数据标签
    # label 时间 ： label_start_time-label_end_time type ：2（下单）
    train_buy = userAll[(userAll['action_time'] >= label_start_time) \
                        & (userAll['action_time'] <= label_end_time) \
                        & (userAll['type'] == 2)][['user_id', 'cate', 'shop_id']].drop_duplicates()
    train_buy['label'] = 1
    # 候选集 时间 ： train_start_time-train_end_time 最近2周有行为的（用户，类目，店铺）
    train_set = userAll[(userAll['action_time'] >= train_start_time) \
                        & (userAll['action_time'] <= train_end_time)][['user_id', 'cate', 'shop_id']].drop_duplicates()
    train_set = train_set.merge(train_buy, on=['user_id', 'cate', 'shop_id'], how='left').fillna(0)
    train_set.to_csv(path, index=False, encoding='utf-8')
    return train_set


def get_test_set_index(test_start_time, test_end_time, path_1, path_2):
    # 候选集 时间 ： test_start_time-test_end_time 最近2周有行为的（用户，类目，店铺）
    test_set = userAll[(userAll['action_time'] >= test_start_time) \
                       & (userAll['action_time'] <= test_end_time)][['user_id', 'cate', 'shop_id']].drop_duplicates()
    test_set.to_csv(path_1, index=False, encoding='utf-8')
    test_set.to_csv(path_2, index=False, encoding='utf-8')
    return test_set


if __name__ == '__main__':
    # 读取行为数据和商品表数据
    jdata_action = pd.read_csv('../data/processsed_data/jdata_action.csv')
    jdata_product = pd.read_csv('../data/processsed_data/jdata_product.csv')
    userAll = jdata_action.merge(jdata_product, on=['sku_id'])

    # 构造不同的训练集和测试集
    get_train_set_label('2018-03-12', '2018-03-25', '2018-03-26', '2018-04-01',
                        '../data/features/features_1_4/train_label_1_4.csv')
    get_train_set_label('2018-03-19', '2018-04-01', '2018-04-02', '2018-04-08',
                        '../data/features/features_2_4/train_label_2_4.csv')
    get_train_set_label('2018-03-26', '2018-04-08', '2018-04-09', '2018-04-15',
                        '../data/features/features_3_4/train_label_3_4.csv')
    get_train_set_label('2018-03-26', '2018-04-08', '2018-04-09', '2018-04-15',
                        '../data/features/features_a_4/train_label_a_4.csv')
    get_test_set_index('2018-04-02', '2018-04-15', '../data/features/features_test/test_index.csv',
                       '../data/features/features_a_test/test_index.csv')
