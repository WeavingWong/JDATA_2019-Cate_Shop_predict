#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/18  10:20
# @Author  : chensw、wangwei
# @File    : make_features.py
# @Describe: 标明文件实现的功能
# @Modify  : 修改的地方

import pandas as pd
import os
import numpy as np


def get_user_features():
    # 统计U类特征
    user_features = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
    user_features = user_features[['user_id']].drop_duplicates()
    # (1)统计个人信息
    Path = '../data/features/features_1_4/userInfo.csv'
    if os.path.exists(Path):
        userInfo = pd.read_csv(Path)
    else:
        userInfo = pd.read_csv('../data/processsed_data/jdata_user.csv')
        userInfo.age.fillna(userInfo.age.median(), inplace=True)
        userInfo.sex.fillna(userInfo.sex.mode()[0], inplace=True)
        userInfo.city_level.fillna(userInfo.city_level.mode()[0], inplace=True)
        userInfo.province.fillna(userInfo.province.mode()[0], inplace=True)
        userInfo.city.fillna(userInfo.city.mode()[0], inplace=True)
        userInfo.county.fillna(userInfo.county.mode()[0], inplace=True)
        print('Check any missing value?\n', userInfo.isnull().any())
        df_user_reg_time = pd.get_dummies(userInfo.user_reg_time, prefix='reg_time')
        df_age = pd.get_dummies(userInfo.age, prefix='age')
        df_sex = pd.get_dummies(userInfo.sex)
        df_city_level = pd.get_dummies(userInfo.city_level, prefix='city_level')
        df_province = pd.get_dummies(userInfo.province, prefix='province')
        df_sex.rename(columns={0: 'female', 1: 'male', -1: 'unknown'}, inplace=True)
        userInfo = pd.concat(
            [userInfo[['user_id', 'user_reg_tm']], df_user_reg_time, df_age, df_sex, df_city_level, df_province],
            axis=1)
        del df_user_reg_time, df_age, df_sex, df_province, df_city_level
        userInfo.drop('user_reg_tm', axis=1, inplace=True)
        # userInfo.to_csv('../data/features/features_1_4/userInfo.csv', index=False, encoding='utf-8')
    user_features = pd.merge(user_features, userInfo, on='user_id', how='left').fillna(0)
    print('userInfo')

    # (2)统计用户的购买频次，用户的平均购买时间间隔, 最后一次购买距离考察周时间
    Path = '../data/features/features_1_4/user_buy_info.csv'
    if os.path.exists(Path):
        user_buy_info = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'action_time', 'type_2']]
        userSub = userSub.sort_values(['user_id', 'action_time'])
        userSub = userSub[(userSub.type_2 == 1)]
        usertmp = userSub.groupby(['user_id'])['action_time'].nunique().reset_index(name='ac_nunique')
        userSub = userSub.merge(usertmp, on='user_id', how='left')
        # print(userSub.head())

        usertmp = userSub[userSub['ac_nunique'] >= 2].copy()
        usertmp['last_time'] = usertmp.groupby(['user_id'])['action_time'].shift(1)
        usertmp['last_time'] = usertmp['last_time'].fillna(-9999)
        usertmp = usertmp[usertmp['last_time'] != -9999]
        usertmp['jiange'] = (pd.to_datetime(usertmp['action_time']) - pd.to_datetime(usertmp['last_time'])).dt.days
        usertmp = usertmp.groupby(['user_id'])['jiange'].mean().reset_index()
        usertmp.columns = ['user_id', 'u_jiange_mean']
        userSub = userSub.merge(usertmp, on='user_id', how='left')
        userSub = userSub[['user_id', 'ac_nunique', 'u_jiange_mean']].drop_duplicates()
        userSub = userSub.fillna(60)

        usertmp = userAll[(userAll['action_time'] <= end_time)]
        usertmp = usertmp[['user_id', 'action_time', 'type_2']].copy()
        usertmp = usertmp[(usertmp.type_2 == 1)]
        usertmp = usertmp.sort_values(['user_id', 'action_time'])
        usertmp = usertmp.drop_duplicates(['user_id'], keep='last')
        usertmp['u_b2_last_day_start'] = (
                pd.to_datetime(label_time_start) - pd.to_datetime(usertmp['action_time'])).dt.days
        usertmp['u_b2_last_day_end'] = (pd.to_datetime(label_time_end) - pd.to_datetime(usertmp['action_time'])).dt.days

        userSub = userSub.merge(usertmp, on='user_id', how='left')
        userSub.drop(['type_2', 'action_time'], axis=1, inplace=True)
        # userSub.to_csv('../data/features/features_1_4/user_buy_info.csv', index=False, encoding='utf-8')
        user_buy_info = userSub.copy()
    user_features = pd.merge(user_features, user_buy_info, on=['user_id'], how='left'). \
        fillna({'ac_nunique': 0, 'u_jiange_mean': 60, 'u_b2_last_day_start': 75, 'u_b2_last_day_end': 75})
    print('user_buy_info')

    # (3)u_n_counts（用户的点击购买行为习惯）
    Path = '../data/features/features_1_4/u_n_counts.csv'
    if os.path.exists(Path):
        u_n_counts = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        typeCount = userSub[["user_id", "cate", "shop_id", "action_time", 'type_2']]
        # 用户考察周前一段时间内做出行为的种类数量
        userSub = typeCount.groupby(['user_id'])['cate'].nunique()
        # 用户考察周前一段时间内做出行为的店铺数量
        userSub = pd.concat([userSub, typeCount.groupby(['user_id'])['shop_id'].nunique()], axis=1)
        # 用户考察周前一段时间内做出行为的天数
        userSub = pd.concat([userSub, typeCount.groupby(['user_id'])['action_time'].nunique()], axis=1)
        userSub.rename(
            columns={'cate': 'u_cate_counts', 'shop_id': 'u_shop_counts', 'action_time': 'u_active_days_in_all'},
            inplace=True)
        userSub.reset_index(inplace=True)
        u_n_counts_in_all = userSub.copy()
        print(u_n_counts_in_all.info())

        typeCount = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        typeCount = typeCount[["user_id", "action_time", 'type_1', 'type_2', 'type_3', 'type_4']]
        # 用户考察周前一周内做出行为的次数
        userSub = typeCount[["user_id", 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id'], as_index=False).sum()
        userSub['u_b_count_in_7'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub['u_b1_count_in_7'] = userSub['type_1']  # 用户在考察周前7天的浏览（1）行为总量计数
        userSub['u_b2_count_in_7'] = userSub['type_2']  # 用户在考察周前7天的下单（2）行为总量计数
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        # 用户考察周前一周内做出行为的天数
        userSub = pd.concat([userSub, typeCount.groupby(['user_id'])['action_time'].nunique()], axis=1)
        userSub.rename(columns={'action_time': 'u_active_days_in_7'}, inplace=True)
        print(userSub.info())
        userSub = pd.merge(u_n_counts_in_all, userSub, how='left', on='user_id').fillna(0)
        u_n_counts = userSub.copy()
        # u_n_counts.to_csv('../data/features/features_1_4/u_n_counts.csv', index=False, encoding='utf-8')
    user_features = pd.merge(user_features, u_n_counts, on='user_id', how='left').fillna(0)
    print('u_n_counts')
    user_features.to_csv('../data/features/features_1_4/user_features.csv', index=False, encoding='utf-8')
    print('user_features!!!!')
    return user_features


def get_ucs_features():
    # step2:构建UCS类特征
    ucs_features = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
    ucs_features = ucs_features[['user_id', 'cate', 'shop_id']].drop_duplicates()
    # (1)统计点击购买转化率，被收藏次数购买转化率, 购买评论转化率等
    Path = '../data/features/features_1_4/ucs_b_rate.csv'
    if os.path.exists(Path):
        ucs_b_rate = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['type_1_ratio'] = np.log1p(userSub['type_2']) - np.log1p(userSub['type_1'])
        userSub['type_3_ratio'] = np.log1p(userSub['type_2']) - np.log1p(userSub['type_3'])
        userSub['type_2_ratio'] = np.log1p(userSub['type_4']) - np.log1p(userSub['type_2'])
        userSub['ucs_b2_rate'] = userSub['type_2'] / (
                userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']).map(
            lambda x: x + 1 if x == 0 else x)
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        ucs_b_rate = userSub.copy()
        # ucs_b_rate.to_csv('../data/features/features_1_4/ucs_b_rate.csv', index=False, encoding='utf-8')
    ucs_features = pd.merge(ucs_features, ucs_b_rate, on=['user_id', 'cate', 'shop_id'], how='left').fillna(0)
    print('ucs_b_rate')

    # (2)ucs_b_count_in_n(n=1/3/5/7/10; 用户对店铺—品类对在考察日前n天的行为总量计数)
    Path = '../data/features/features_1_4/ucs_b_count_in_n.csv'
    if os.path.exists(Path):
        ucs_b_count_in_n = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] > end_time_1) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['ucs_b_count_in_1'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        ucs_b_count_in_1 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['ucs_b_count_in_3'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        ucs_b_count_in_3 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['ucs_b_count_in_5'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        ucs_b_count_in_5 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['ucs_b_count_in_7'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        ucs_b_count_in_7 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_10) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['ucs_b_count_in_10'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        ucs_b_count_in_10 = userSub.copy()

        ucs_b_count_in_n = pd.merge(ucs_b_count_in_10, ucs_b_count_in_7, on=['user_id', 'cate', 'shop_id'],
                                    how='left').fillna(0)
        ucs_b_count_in_n = pd.merge(ucs_b_count_in_n, ucs_b_count_in_5, on=['user_id', 'cate', 'shop_id'],
                                    how='left').fillna(0)
        ucs_b_count_in_n = pd.merge(ucs_b_count_in_n, ucs_b_count_in_3, on=['user_id', 'cate', 'shop_id'],
                                    how='left').fillna(0)
        ucs_b_count_in_n = pd.merge(ucs_b_count_in_n, ucs_b_count_in_1, on=['user_id', 'cate', 'shop_id'],
                                    how='left').fillna(0)
        # print(ucs_b_count_in_n.info())
        # print(ucs_b_count_in_n.head())
        # ucs_b_count_in_n.to_csv('../data/features/features_1_4/ucs_b_count_in_n.csv', index=False, encoding='utf-8')
    ucs_features = pd.merge(ucs_features, ucs_b_count_in_n, on=['user_id', 'cate', 'shop_id'], how='left').fillna(0)
    print('ucs_b_count_in_n')

    # (3)ucs_bi_count_in_n(n=3/5/7;i=1/2/3/4/5; 用户在考察日前n天的各类行为总量计数，)
    Path = '../data/features/features_1_4/ucs_bi_count_in_n.csv'
    if os.path.exists(Path):
        ucs_bi_count_in_n = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] > end_time_1) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['ucs_b1_count_in_1'] = userSub['type_1']  # 用户在考察周前1天的浏览（1）行为总量计数
        userSub['ucs_b2_count_in_1'] = userSub['type_2']  # 用户在考察周前1天的下单（2）行为总量计数
        userSub.drop(['type_1'], axis=1, inplace=True)
        userSub.drop(['type_2'], axis=1, inplace=True)
        ucs_bi_count_in_1 = userSub.copy()
        # print(ucs_bi_count_in_1.info())
        # print(ucs_bi_count_in_1.ucs_b1_count_in_3.max())

        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['ucs_b1_count_in_3'] = userSub['type_1']  # 用户在考察周前3天的浏览（1）行为总量计数
        userSub['ucs_b2_count_in_3'] = userSub['type_2']  # 用户在考察周前3天的下单（2）行为总量计数
        userSub['ucs_b3_count_in_3'] = userSub['type_3']  # 用户在考察周前3天的关注（3）行为总量计数
        userSub['ucs_b4_count_in_3'] = userSub['type_4']  # 用户在考察周前3天的评论（4）行为总量计数
        userSub.drop(['type_1'], axis=1, inplace=True)
        userSub.drop(['type_2'], axis=1, inplace=True)
        userSub.drop(['type_3'], axis=1, inplace=True)
        userSub.drop(['type_4'], axis=1, inplace=True)
        ucs_bi_count_in_3 = userSub.copy()
        # print(ucs_bi_count_in_3.info())
        # print(ucs_bi_count_in_3.ucs_b1_count_in_3.max())

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['ucs_b1_count_in_5'] = userSub['type_1']  # 用户在考察周前5天的浏览（1）行为总量计数
        userSub['ucs_b2_count_in_5'] = userSub['type_2']  # 用户在考察周前5天的下单（2）行为总量计数
        userSub['ucs_b3_count_in_5'] = userSub['type_3']  # 用户在考察周前5天的关注（3）行为总量计数
        userSub['ucs_b4_count_in_5'] = userSub['type_4']  # 用户在考察周前5天的评论（4）行为总量计数
        userSub.drop(['type_1'], axis=1, inplace=True)
        userSub.drop(['type_2'], axis=1, inplace=True)
        userSub.drop(['type_3'], axis=1, inplace=True)
        userSub.drop(['type_4'], axis=1, inplace=True)
        ucs_bi_count_in_5 = userSub.copy()
        # print(ucs_bi_count_in_5.info())
        # print(ucs_bi_count_in_5.ucs_b3_count_in_5.max())

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['ucs_b1_count_in_7'] = userSub['type_1']  # 用户在考察周前7天的浏览（1）行为总量计数
        userSub['ucs_b2_count_in_7'] = userSub['type_2']  # 用户在考察周前7天的下单（2）行为总量计数
        userSub['ucs_b3_count_in_7'] = userSub['type_3']  # 用户在考察周前7天的关注（3）行为总量计数
        userSub['ucs_b4_count_in_7'] = userSub['type_4']  # 用户在考察周前7天的评论（4）行为总量计数
        userSub.drop(['type_1'], axis=1, inplace=True)
        userSub.drop(['type_2'], axis=1, inplace=True)
        userSub.drop(['type_3'], axis=1, inplace=True)
        userSub.drop(['type_4'], axis=1, inplace=True)
        ucs_bi_count_in_7 = userSub.copy()
        # print(ucs_bi_count_in_7.info())
        # print(ucs_bi_count_in_7.ucs_b2_count_in_7.max())

        userSub = userAll[(userAll['action_time'] >= end_time_10) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['ucs_b1_count_in_10'] = userSub['type_1']  # 用户在考察周前10天的浏览（1）行为总量计数
        userSub['ucs_b2_count_in_10'] = userSub['type_2']  # 用户在考察周前10天的下单（2）行为总量计数
        userSub['ucs_b3_count_in_10'] = userSub['type_3']  # 用户在考察周前10天的关注（3）行为总量计数
        userSub['ucs_b4_count_in_10'] = userSub['type_4']  # 用户在考察周前10天的评论（4）行为总量计数
        userSub.drop(['type_1'], axis=1, inplace=True)
        userSub.drop(['type_2'], axis=1, inplace=True)
        userSub.drop(['type_3'], axis=1, inplace=True)
        userSub.drop(['type_4'], axis=1, inplace=True)
        ucs_bi_count_in_10 = userSub.copy()
        # print(ucs_bi_count_in_10.info())
        # print(ucs_bi_count_in_10.ucs_b2_count_in_10.max())

        ucs_bi_count_in_n = pd.merge(ucs_bi_count_in_10, ucs_bi_count_in_7, on=['user_id', 'cate', 'shop_id'],
                                     how='left').fillna(0)
        ucs_bi_count_in_n = pd.merge(ucs_bi_count_in_n, ucs_bi_count_in_5, on=['user_id', 'cate', 'shop_id'],
                                     how='left').fillna(0)
        ucs_bi_count_in_n = pd.merge(ucs_bi_count_in_n, ucs_bi_count_in_3, on=['user_id', 'cate', 'shop_id'],
                                     how='left').fillna(0)
        ucs_bi_count_in_n = pd.merge(ucs_bi_count_in_n, ucs_bi_count_in_1, on=['user_id', 'cate', 'shop_id'],
                                     how='left').fillna(0)
        # ucs_bi_count_in_n.to_csv('../data/features/features_1_4/ucs_bi_count_in_n.csv', index=False, encoding='utf-8')
    ucs_features = pd.merge(ucs_features, ucs_bi_count_in_n, on=['user_id', 'cate', 'shop_id'], how='left').fillna(0)
    print('ucs_bi_count_in_n')

    # (4)ucs_b2_diff_day（用户的点击购买平均时差，反映了用户的购买决策时间习惯）
    Path = '../data/features/features_1_4/ucs_b2_diff_day.csv'
    if os.path.exists(Path):
        ucs_b2_diff_day = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub = userSub[['action_time', 'user_id', 'cate', 'shop_id', 'type_1', 'type_2']]
        userSub.drop_duplicates(inplace=True)
        userSub = userSub.sort_values(by=['user_id', 'cate', 'shop_id', 'action_time'], axis=0, ascending=False)
        # print(userSub.head(20))

        usertmp1 = userSub[userSub['type_1'] == 1].drop(['type_2'], axis=1)
        usertmp1 = usertmp1.drop_duplicates(['user_id', 'cate', 'shop_id', 'type_1'], keep='last', inplace=False)
        usertmp1.rename(columns={'action_time': 'action_time_1'}, inplace=True)
        usertmp2 = userSub[userSub['type_2'] == 1].drop(['type_1'], axis=1)
        usertmp2 = usertmp2.drop_duplicates(['user_id', 'cate', 'shop_id', 'type_2'], keep='first', inplace=False)
        usertmp2.rename(columns={'action_time': 'action_time_2'}, inplace=True)
        # print(usertmp1.head())
        # print(usertmp2.head())
        usertmp = pd.merge(usertmp1, usertmp2, how='inner', on=['user_id', 'cate', 'shop_id'])
        # print(usertmp.head(20))

        userSub = userSub[userSub['type_2'] == 1].drop(['type_1'], axis=1)
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'])['action_time'].nunique().reset_index(
            name='ucs_ac_nunique')
        userSub = pd.merge(usertmp, userSub, on=['user_id', 'cate', 'shop_id'], how='left')

        userSub['ucs_b2_diff_day'] = (
                pd.to_datetime(userSub['action_time_2']) - pd.to_datetime(userSub['action_time_1'])).dt.days
        # print(usertmp.head(20))
        userSub['ucs_b2_diff_day'] = userSub['ucs_b2_diff_day'] / userSub['ucs_ac_nunique']
        ucs_b2_diff_day = userSub[['user_id', 'cate', 'shop_id', 'ucs_ac_nunique', 'ucs_b2_diff_day']]
        # ucs_b2_diff_day.to_csv('../data/features/features_1_4/ucs_b2_diff_day.csv', index=False, encoding='utf-8')
    ucs_features = pd.merge(ucs_features, ucs_b2_diff_day, on=['user_id', 'cate', 'shop_id'], how='left').fillna(60)
    print('ucs_b2_diff_day')

    # (5)ucs_b1_last_day（用户的最近的一次点击距离考察周的时间）
    Path = '../data/features/features_1_4/ucs_b1_last_day.csv'
    if os.path.exists(Path):
        usertmp = pd.read_csv(Path)
    else:
        usertmp = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
        usertmp = usertmp[['user_id', 'cate', 'shop_id', 'action_time', 'type_1']]
        usertmp = usertmp[(usertmp.type_1 == 1)]

        usertmp = usertmp.sort_values(['user_id', 'cate', 'shop_id', 'action_time'])
        # print(usertmp.head(20))
        usertmp = usertmp.drop_duplicates(['user_id', 'cate', 'shop_id'], keep='last')
        usertmp['ucs_b1_last_min_day'] = (
                pd.to_datetime(label_time_start) - pd.to_datetime(usertmp['action_time'])).dt.days
        usertmp['ucs_b1_last_max_day'] = (
                pd.to_datetime(label_time_end) - pd.to_datetime(usertmp['action_time'])).dt.days
        # print(usertmp.head(20))
        usertmp = usertmp[['user_id', 'cate', 'shop_id', 'ucs_b1_last_min_day', 'ucs_b1_last_max_day']]
        # usertmp.to_csv('../data/features/features_1_4/ucs_b1_last_day.csv', index=False, encoding='utf-8')
    ucs_features = pd.merge(ucs_features, usertmp, on=['user_id', 'cate', 'shop_id'], how='left').fillna(60)
    print('ucs_b1_last_day')

    # (6)ucs_active_days（用户的点击购买行为习惯）
    Path = '../data/features/features_1_4/ucs_active_days.csv'
    if os.path.exists(Path):
        ucs_active_days = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
        userSub = userSub[["user_id", "cate", "shop_id", "action_time"]]
        # 用户-品类-店铺对考察周前一段时间内做出行为的天数
        userSub = userSub.groupby(["user_id", "cate", "shop_id"])['action_time'].nunique().reset_index()
        userSub.rename(columns={'action_time': 'ucs_active_days'}, inplace=True)
        ucs_active_days = userSub.copy()
        # print(ucs_active_days.info())
        # print(ucs_active_days.head())
        # ucs_active_days.to_csv('../data/features/features_1_4/ucs_active_days.csv', index=False, encoding='utf-8')
    ucs_features = pd.merge(ucs_features, ucs_active_days, on=['user_id', 'cate', 'shop_id'], how='left').fillna(0)
    print('ucs_active_days')

    # (7)ucs_b_count_rank_in_n_in_u(用户-类别_店铺对的行为在用户所有商品中的排序  反映了user_id对item_category+shop_id的行为偏好)
    Path = '../data/features/features_1_4/ucs_b_count_rank_in_n_in_u.csv'
    if os.path.exists(Path):
        ucs_b_count_rank_in_n_in_u = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] > end_time_1) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['b_count'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        userSub['ucs_b_count_rank_in_1_in_u'] = userSub['b_count'].groupby(userSub['user_id']).rank(ascending=0,
                                                                                                    method='dense')
        userSub.drop('b_count', axis=1, inplace=True)
        ucs_b_count_rank_in_1_in_u = userSub.copy()
        # print(ucs_b_count_rank_in_1_in_u.info())
        # print(ucs_b_count_rank_in_1_in_u.head(20))

        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['b_count'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        userSub['ucs_b_count_rank_in_3_in_u'] = userSub['b_count'].groupby(userSub['user_id']).rank(ascending=0,
                                                                                                    method='dense')
        userSub.drop('b_count', axis=1, inplace=True)
        ucs_b_count_rank_in_3_in_u = userSub.copy()
        # print(ucs_b_count_rank_in_3_in_u.info())
        # print(ucs_b_count_rank_in_3_in_u.head(20))

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['b_count'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        userSub['ucs_b_count_rank_in_5_in_u'] = userSub['b_count'].groupby(userSub['user_id']).rank(ascending=0,
                                                                                                    method='dense')
        userSub.drop('b_count', axis=1, inplace=True)
        ucs_b_count_rank_in_5_in_u = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        userSub['b_count'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        userSub['ucs_b_count_rank_in_7_in_u'] = userSub['b_count'].groupby(userSub['user_id']).rank(ascending=0,
                                                                                                    method='dense')
        userSub.drop('b_count', axis=1, inplace=True)
        ucs_b_count_rank_in_7_in_u = userSub.copy()
        # 没有的置为1000(没有竞争力的一个靠后排名)
        ucs_b_count_rank_in_n_in_u = pd.merge(ucs_b_count_rank_in_7_in_u, ucs_b_count_rank_in_5_in_u, how='left',
                                              on=['user_id', 'cate', 'shop_id']).fillna(1000)
        ucs_b_count_rank_in_n_in_u = pd.merge(ucs_b_count_rank_in_n_in_u, ucs_b_count_rank_in_3_in_u, how='left',
                                              on=['user_id', 'cate', 'shop_id']).fillna(1000)
        ucs_b_count_rank_in_n_in_u = pd.merge(ucs_b_count_rank_in_n_in_u, ucs_b_count_rank_in_1_in_u, how='left',
                                              on=['user_id', 'cate', 'shop_id']).fillna(1000)

        # ucs_b_count_rank_in_n_in_u.to_csv('../data/features/features_1_4/ucs_b_count_rank_in_n_in_u.csv', index=False, encoding='utf-8')
        # print(ucs_b_count_rank_in_n_in_u.info())
        # print(ucs_b_count_rank_in_n_in_u.head())
    ucs_features = pd.merge(ucs_features, ucs_b_count_rank_in_n_in_u, on=['user_id', 'cate', 'shop_id'],
                            how='left').fillna(0)
    print('ucs_b_count_rank_in_n_in_u')

    # (8)统计用户-店铺-品类的购买频次平均购买时间间隔, 最后一次购买距离考察周时间
    Path = '../data/features/features_1_4/ucs_buy_info.csv'
    if os.path.exists(Path):
        ucs_buy_info = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'cate', 'type_2', 'action_time']].copy()
        userSub = userSub.sort_values(['user_id', 'shop_id', 'cate', 'action_time'])
        userSub = userSub[(userSub.type_2 == 1)]
        usertmp = userSub.groupby(['user_id', 'shop_id', 'cate'])['action_time'].nunique().reset_index(
            name='ucs_ac_nunique')
        userSub = userSub.merge(usertmp, on=['user_id', 'shop_id', 'cate'], how='left')
        # print(userSub.head())

        usertmp = userSub[userSub['ucs_ac_nunique'] >= 2].copy()
        usertmp['last_time'] = usertmp.groupby(['user_id', 'shop_id', 'cate'])['action_time'].shift(1)
        usertmp['last_time'] = usertmp['last_time'].fillna(-9999)
        usertmp = usertmp[usertmp['last_time'] != -9999]
        usertmp['jiange'] = (pd.to_datetime(usertmp['action_time']) - pd.to_datetime(usertmp['last_time'])).dt.days
        usertmp = usertmp.groupby(['user_id', 'shop_id', 'cate'])['jiange'].mean().reset_index()
        usertmp.columns = ['user_id', 'shop_id', 'cate', 'ucs_jiange_mean']
        userSub = userSub.merge(usertmp, on=['user_id', 'shop_id', 'cate'], how='left')
        userSub = userSub[['user_id', 'shop_id', 'cate', 'ucs_ac_nunique', 'ucs_jiange_mean']].drop_duplicates()
        userSub = userSub.fillna(60)

        usertmp = userAll[(userAll['action_time'] <= end_time)]
        usertmp = usertmp[['user_id', 'shop_id', 'cate', 'action_time', 'type_2']].copy()
        usertmp = usertmp[(usertmp.type_2 == 1)]
        usertmp = usertmp.sort_values(['user_id', 'shop_id', 'cate', 'action_time'])
        usertmp = usertmp.drop_duplicates(['user_id', 'shop_id', 'cate'], keep='last')
        usertmp['ucs_b2_last_day_start'] = (
                pd.to_datetime(label_time_start) - pd.to_datetime(usertmp['action_time'])).dt.days
        usertmp['ucs_b2_last_day_end'] = (
                pd.to_datetime(label_time_end) - pd.to_datetime(usertmp['action_time'])).dt.days

        userSub = userSub.merge(usertmp, on=['user_id', 'shop_id', 'cate'], how='left')
        userSub.drop(['type_2', 'action_time'], axis=1, inplace=True)
        # userSub.to_csv('../data/features/features_1_4/ucs_buy_info.csv', index=False, encoding='utf-8')
        ucs_buy_info = userSub.copy()
    ucs_features = pd.merge(ucs_features, ucs_buy_info, on=['user_id', 'shop_id', 'cate'], how='left'). \
        fillna({'ucs_ac_nunique': 0, 'ucs_jiange_mean': 60, 'ucs_b2_last_day_start': 75, 'ucs_b2_last_day_end': 75})
    print('ucs_buy_info')

    print('-->Counting total numbers of shop, cate and action_days in various Users<-- are finished...')

    filePath = '../data/features/features_1_4/ucs_features.csv'
    ucs_features.to_csv(filePath, index=False, encoding='utf-8')
    print('ucs_features!!!!')
    return ucs_features


def get_cateInfo_features():
    # step4:构建C(cate)类特征
    cate_features = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
    cate_features = cate_features[['cate']].drop_duplicates()
    # (1)c_b2_rate(类别的点击购买转化率  反映了item_category的购买决策操作特点)
    Path = '../data/features/features_1_4/c_b2_rate.csv'
    if os.path.exists(Path):
        c_b2_rate = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub.drop_duplicates(inplace=True)
        userSub = userSub[['cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['cate'], as_index=False).sum()
        userSub['c_type_1_ratio'] = np.log1p(userSub['type_2']) - np.log1p(userSub['type_1'])
        userSub['c_type_3_ratio'] = np.log1p(userSub['type_2']) - np.log1p(userSub['type_3'])
        userSub['c_type_2_ratio'] = np.log1p(userSub['type_4']) - np.log1p(userSub['type_2'])
        userSub['c_b2_rate'] = userSub['type_2'] / (
                userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']).map(
            lambda x: x if x != 0 else x + 1)
        # print(userSub.head())
        c_b2_rate = userSub[['cate', 'c_b2_rate', 'c_type_1_ratio', 'c_type_2_ratio', 'c_type_3_ratio']]
        # print(c_b2_rate.info())
        # print(c_b2_rate.head())
        # c_b2_rate.to_csv('../data/features/features_1_4/c_b2_rate.csv', index=False, encoding='utf-8')
    cate_features = pd.merge(cate_features, c_b2_rate, on='cate', how='left').fillna(0)
    print('c_b2_rate')

    # (2)c_all_count(品类下商品，品牌，店铺的数量)
    Path = '../data/features/features_1_4/c_all_count.csv'
    if os.path.exists(Path):
        c_all_count = pd.read_csv(Path)
    else:
        userSub = jdata_product[['sku_id', 'cate', 'brand', 'shop_id']].copy()
        usertmp1 = userSub.groupby(['cate'])['brand'].nunique().reset_index()
        usertmp2 = userSub.groupby(['cate'])['sku_id'].nunique().reset_index()
        userSub = userSub.groupby(['cate'])['shop_id'].nunique().reset_index()
        userSub = pd.merge(userSub, usertmp1, how='left', on='cate')
        userSub = pd.merge(userSub, usertmp2, how='left', on='cate')
        print(userSub.head())
        userSub.rename(columns={'sku_id': 'c_item_count', 'brand': 'c_brand_count', 'shop_id': 'c_shop_count'},
                       inplace=True)
        c_all_count = userSub.copy()
        # c_all_count.to_csv('../data/features/features_1_4/c_all_count.csv', index=False, encoding='utf-8')
    cate_features = pd.merge(cate_features, c_all_count, on='cate', how='left').fillna(0)
    print('c_all_count')

    # (3)c_new_item_count(品类下新品的数量)
    Path = '../data/features/features_1_4/c_new_item_count.csv'
    if os.path.exists(Path):
        c_new_item_count = pd.read_csv(Path)
    else:
        userSub = jdata_product[['cate', 'market_tm']].copy()
        df_market_tm = pd.get_dummies(userSub.market_tm, prefix='c_market_tm')
        userSub = pd.concat([userSub.cate, df_market_tm], axis=1)
        del df_market_tm
        userSub = userSub.groupby(['cate'], as_index=False).sum()
        c_new_item_count = userSub.copy()
        # c_new_item_count.to_csv('../data/features/features_1_4/c_new_item_count.csv', index=False, encoding='utf-8')
    cate_features = pd.merge(cate_features, c_new_item_count, on='cate', how='left').fillna(0)
    print('c_new_item_count')

    # (4)c_u_count_in_n(类别在考察日前n天的用户总数计数   反映了item_category的热度（用户覆盖性）)
    Path = '../data/features/features_1_4/c_u_count_in_n.csv'
    if os.path.exists(Path):
        c_u_count_in_n = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate']]
        userSub = userSub.drop_duplicates()  # 去除重复的用户
        userSub = userSub.drop('user_id', axis=1, inplace=False)
        # print(userSub.info())
        userSub = userSub[['cate']]
        userSub['c_u_count_in_3'] = 1
        # print(userSub.head())
        userSub = userSub.groupby(['cate'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.c_u_count_in_1.max())
        c_u_count_in_3 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub.drop_duplicates()
        userSub = userSub[['cate']]
        userSub['c_u_count_in_5'] = 1
        userSub = userSub.groupby(['cate'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.c_u_count_in_5.max())
        c_u_count_in_5 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub.drop_duplicates()
        userSub = userSub[['cate']]
        userSub['c_u_count_in_7'] = 1
        userSub = userSub.groupby(['cate'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.c_u_count_in_7.max())
        c_u_count_in_7 = userSub.copy()

        c_u_count_in_n = pd.merge(c_u_count_in_7, c_u_count_in_5, on=['cate'], how='left').fillna(0)
        c_u_count_in_n = pd.merge(c_u_count_in_n, c_u_count_in_3, on=['cate'], how='left').fillna(0)
        # print(c_u_count_in_n.info())
        # print(c_u_count_in_n.head())
        # c_u_count_in_n.to_csv('../data/features/features_1_4/c_u_count_in_n.csv', index=False, encoding='utf-8')
    cate_features = pd.merge(cate_features, c_u_count_in_n, on='cate', how='left').fillna(0)
    print('c_u_count_in_n')

    # (5)c_bi_count_in_n(类别在考察日前n天的各项行为计数  反映了item_category的热度（用户操作吸引），包含着item_category产生的购买习惯特点)
    Path = '../data/features/features_1_4/c_bi_count_in_n.csv'
    if os.path.exists(Path):
        c_bi_count_in_n = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] > end_time_1) & (userAll['action_time'] <= end_time)]
        userSub = userSub.drop_duplicates()
        userSub = userSub[['cate', 'type_1']]
        userSub = userSub.groupby(['cate'], as_index=False).sum()
        # print(userSub.head())
        userSub.rename(columns={'type_1': 'c_b1_count_in_1'}, inplace=True)
        # print(userSub.info())
        # print(userSub.head())
        c_bi_count_in_1 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        userSub = userSub.drop_duplicates()
        userSub = userSub[['cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['cate'], as_index=False).sum()
        # print(userSub.head())
        userSub.rename(columns={'type_1': 'c_b1_count_in_3', 'type_2': 'c_b2_count_in_3', 'type_3': 'c_b3_count_in_3',
                                'type_4': 'c_b4_count_in_3'}, inplace=True)
        # print(userSub.info())
        # print(userSub.head())
        c_bi_count_in_3 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub.drop_duplicates()
        userSub = userSub[['cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['cate'], as_index=False).sum()
        userSub.rename(columns={'type_1': 'c_b1_count_in_5', 'type_2': 'c_b2_count_in_5', 'type_3': 'c_b3_count_in_5',
                                'type_4': 'c_b4_count_in_5'}, inplace=True)
        c_bi_count_in_5 = userSub.copy()
        # print(c_bi_count_in_5.info())
        # print(c_bi_count_in_5.head())

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub.drop_duplicates()
        userSub = userSub[['cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['cate'], as_index=False).sum()
        userSub.rename(columns={'type_1': 'c_b1_count_in_7', 'type_2': 'c_b2_count_in_7', 'type_3': 'c_b3_count_in_7',
                                'type_4': 'c_b4_count_in_7'}, inplace=True)
        c_bi_count_in_7 = userSub.copy()
        # print(c_bi_count_in_7.info())
        # print(c_bi_count_in_7.head())

        c_bi_count_in_n = pd.merge(c_bi_count_in_7, c_bi_count_in_5, on=['cate'], how='left').fillna(0)
        c_bi_count_in_n = pd.merge(c_bi_count_in_n, c_bi_count_in_3, on=['cate'], how='left').fillna(0)
        c_bi_count_in_n = pd.merge(c_bi_count_in_n, c_bi_count_in_1, on=['cate'], how='left').fillna(0)
        # print(c_bi_count_in_n.info())
        # print(c_bi_count_in_n.head())
        # c_bi_count_in_n.to_csv('../data/features/features_1_4/c_bi_count_in_n.csv', index=False, encoding='utf-8')
    cate_features = pd.merge(cate_features, c_bi_count_in_n, on='cate', how='left').fillna(0)
    print('c_bi_count_in_n')

    # (6)c_b_count_in_n(类别在考察日前n天的行为总数计数   反映了item_category的热度（用户停留性）)
    Path = '../data/features/features_1_4/c_b_count_in_n.csv'
    if os.path.exists(Path):
        c_b_count_in_n = pd.read_csv(Path)
    else:
        userSub = c_bi_count_in_n.copy()
        userSub['c_b_count_in_3'] = userSub['c_b1_count_in_3'] + userSub['c_b2_count_in_3'] + userSub[
            'c_b3_count_in_3'] + \
                                    userSub['c_b4_count_in_3']
        userSub['c_b_count_in_5'] = userSub['c_b1_count_in_5'] + userSub['c_b2_count_in_5'] + userSub[
            'c_b3_count_in_5'] + \
                                    userSub['c_b4_count_in_5']
        userSub['c_b_count_in_7'] = userSub['c_b1_count_in_7'] + userSub['c_b2_count_in_7'] + userSub[
            'c_b3_count_in_7'] + \
                                    userSub['c_b4_count_in_7']
        userSub = userSub[['cate', 'c_b_count_in_3', 'c_b_count_in_5', 'c_b_count_in_7']]
        # print(userSub.info())
        # print(userSub.head())
        c_b_count_in_n = userSub.copy()
        # c_b_count_in_n.to_csv('../data/features/features_1_4/c_b_count_in_n.csv', index=False, encoding='utf-8')
    cate_features = pd.merge(cate_features, c_b_count_in_n, on='cate', how='left').fillna(0)
    print('c_b_count_in_n')

    # (7)c_b2_count(类别被购买的频次)
    Path = '../data/features/features_1_4/c_b2_count.csv'
    if os.path.exists(Path):
        c_b2_count = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub = userSub[['cate', 'type_2']].copy()
        userSub = userSub[(userSub.type_2 == 1)]
        c_b2_count = userSub.groupby(['cate'])['type_2'].count().reset_index(name='c_b2_count')
        # c_b2_count.to_csv('../data/features/features_1_4/c_b2_count.csv', index=False, encoding='utf-8')
    cate_features = pd.merge(cate_features, c_b2_count, on='cate', how='left').fillna(0)
    print('c_b2_count')

    # (8)品类的评论
    Path = '../data/features/features_1_4/c_comments_counts.csv'
    if os.path.exists(Path):
        c_comments_counts = pd.read_csv(Path)
    else:
        user_comment = pd.read_csv('../data/processsed_data/jdata_comment.csv')
        user_comment = user_comment[(user_comment['dt'] >= start_time) & (user_comment['dt'] <= end_time)]
        user_comment = user_comment[['sku_id', 'comments', 'good_comments', 'bad_comments']]
        user_comment = user_comment.groupby(['sku_id'], as_index=False).sum()
        userSub = pd.merge(userAll, user_comment, how='left', on='sku_id').fillna(0)
        userSub = userSub[['cate', 'comments', 'good_comments', 'bad_comments']]
        userSub = userSub.groupby(['cate'], as_index=False).sum()
        userSub.rename(
            columns={'comments': 'c_comments', 'good_comments': 'c_good_comments', 'bad_comments': 'c_bad_comments'},
            inplace=True)
        # userSub.to_csv('../data/features/features_1_4/c_comments_counts.csv', index=False, encoding='utf-8')
        c_comments_counts = userSub[['cate', 'c_comments', 'c_good_comments', 'c_bad_comments']]

    cate_features = pd.merge(cate_features, c_comments_counts, on='cate', how='left').fillna(0)
    filePath = '../data/features/features_1_4/cate_features.csv'
    cate_features.to_csv(filePath, index=False, encoding='utf-8')
    print('cateInfo')
    return cate_features


def get_shopInfo_features():
    # step5:构建S(shop)类特征
    shop_features = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
    shop_features = shop_features[['shop_id']].drop_duplicates()
    # (1)s_b2_rate(店铺的点击购买转化率  反映了店铺的购买决策操作特点)
    Path = '../data/features/features_1_4/s_b2_rate.csv'
    if os.path.exists(Path):
        s_b2_rate = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
        userSub.drop_duplicates(inplace=True)
        userSub = userSub[['shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['shop_id'], as_index=False).sum()
        userSub['s_type_1_ratio'] = np.log1p(userSub['type_2']) - np.log1p(userSub['type_1'])
        userSub['s_type_3_ratio'] = np.log1p(userSub['type_2']) - np.log1p(userSub['type_3'])
        userSub['s_type_2_ratio'] = np.log1p(userSub['type_4']) - np.log1p(userSub['type_2'])
        userSub['s_b2_rate'] = userSub['type_2'] / (
                userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']).map(
            lambda x: x if x != 0 else x + 1)
        # print(userSub.head())
        s_b2_rate = userSub[['shop_id', 's_b2_rate', 's_type_1_ratio', 's_type_2_ratio', 's_type_3_ratio']]
        # print(s_b2_rate.info())
        # print(s_b2_rate.head())
        # s_b2_rate.to_csv('../data/features/features_1_4/s_b2_rate.csv', index=False, encoding='utf-8')
    shop_features = pd.merge(shop_features, s_b2_rate, on='shop_id', how='left').fillna(0)
    print('s_b2_rate')

    # (2)s_all_count(店铺下品类,商品的个数)
    Path = '../data/features/features_1_4/s_all_count.csv'
    if os.path.exists(Path):
        s_all_count = pd.read_csv(Path)
    else:
        userSub = jdata_product[['shop_id', 'sku_id', 'cate']].copy()
        usertmp1 = userSub.groupby(['shop_id'])['sku_id'].nunique().reset_index()
        userSub = userSub.groupby(['shop_id'])['cate'].nunique().reset_index()
        userSub = pd.merge(userSub, usertmp1, how='left', on='shop_id')
        userSub.rename(columns={'sku_id': 's_item_count', 'cate': 's_cate_count'}, inplace=True)
        s_all_count = userSub.copy()
        # s_all_count.to_csv('../data/features/features_1_4/s_all_count.csv', index=False, encoding='utf-8')
    shop_features = pd.merge(shop_features, s_all_count, on='shop_id', how='left').fillna(0)
    print('s_all_count')

    # (3)s_new_item_count(店铺下新品的数量)
    Path = '../data/features/features_1_4/s_new_item_count.csv'
    if os.path.exists(Path):
        s_new_item_count = pd.read_csv(Path)
    else:
        userSub = jdata_product[['shop_id', 'market_tm']].copy()
        df_market_tm = pd.get_dummies(userSub.market_tm, prefix='s_market_tm')
        userSub = pd.concat([userSub.shop_id, df_market_tm], axis=1)
        del df_market_tm
        userSub = userSub.groupby(['shop_id'], as_index=False).sum()
        s_new_item_count = userSub.copy()
        # s_new_item_count.to_csv('../data/features/features_1_4/s_new_item_count.csv', index=False, encoding='utf-8')
    shop_features = pd.merge(shop_features, s_new_item_count, on='shop_id', how='left').fillna(0)
    print('s_new_item_count')

    # (2)s_u_count_in_n(店铺在考察日前n天的用户总数计数,反映了shop的热度（用户覆盖性）)
    Path = '../data/features/features_1_4/s_u_count_in_n.csv'
    if os.path.exists(Path):
        s_u_count_in_n = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        # print(userSub.info())
        userSub = userSub[['user_id', 'shop_id']]
        userSub = userSub.drop_duplicates()  # 去除重复的用户
        userSub = userSub.drop('user_id', axis=1, inplace=False)
        # print(userSub.info())
        userSub['s_u_count_in_3'] = 1
        # print(userSub.head())
        userSub = userSub.groupby(['shop_id'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.s_u_count_in_3.max())
        s_u_count_in_3 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub.drop_duplicates()
        userSub = userSub[['shop_id']]
        userSub['s_u_count_in_5'] = 1
        userSub = userSub.groupby(['shop_id'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.s_u_count_in_5.max())
        s_u_count_in_5 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub.drop_duplicates()
        userSub = userSub[['shop_id']]
        userSub['s_u_count_in_7'] = 1
        userSub = userSub.groupby(['shop_id'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.s_u_count_in_7.max())
        s_u_count_in_7 = userSub.copy()

        s_u_count_in_n = pd.merge(s_u_count_in_7, s_u_count_in_5, on=['shop_id'], how='left').fillna(0)
        s_u_count_in_n = pd.merge(s_u_count_in_n, s_u_count_in_3, on=['shop_id'], how='left').fillna(0)
        # print(s_u_count_in_n.info())
        # print(s_u_count_in_n.head())
        # s_u_count_in_n.to_csv('../data/features/features_1_4/s_u_count_in_n.csv', index=False, encoding='utf-8')
    shop_features = pd.merge(shop_features, s_u_count_in_n, on='shop_id', how='left').fillna(0)

    # (3)s_bi_count_in_n(店铺在考察日前n天的各项行为计数  反映了店铺的热度（用户操作吸引），包含着店铺产生的购买习惯特点)
    Path = '../data/features/features_1_4/s_bi_count_in_n.csv'
    if os.path.exists(Path):
        s_bi_count_in_n = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] > end_time_1) & (userAll['action_time'] <= end_time)]
        userSub = userSub.drop_duplicates()
        userSub = userSub[['shop_id', 'type_1']]
        userSub = userSub.groupby(['shop_id'], as_index=False).sum()
        # print(userSub.head())
        userSub.rename(columns={'type_1': 's_b1_count_in_1'}, inplace=True)
        # print(userSub.info())
        # print(userSub.head())
        s_bi_count_in_1 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        userSub = userSub.drop_duplicates()
        userSub = userSub[['shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['shop_id'], as_index=False).sum()
        # print(userSub.head())
        userSub.rename(columns={'type_1': 's_b1_count_in_3', 'type_2': 's_b2_count_in_3', 'type_3': 's_b3_count_in_3',
                                'type_4': 's_b4_count_in_3'}, inplace=True)
        # print(userSub.info())
        # print(userSub.head())
        s_bi_count_in_3 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub.drop_duplicates()
        userSub = userSub[['shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['shop_id'], as_index=False).sum()
        userSub.rename(columns={'type_1': 's_b1_count_in_5', 'type_2': 's_b2_count_in_5', 'type_3': 's_b3_count_in_5',
                                'type_4': 's_b4_count_in_5'}, inplace=True)
        s_bi_count_in_5 = userSub.copy()
        # print(s_bi_count_in_5.info())
        # print(s_bi_count_in_5.head())

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub.drop_duplicates()
        userSub = userSub[['shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['shop_id'], as_index=False).sum()
        userSub.rename(columns={'type_1': 's_b1_count_in_7', 'type_2': 's_b2_count_in_7', 'type_3': 's_b3_count_in_7',
                                'type_4': 's_b4_count_in_7'}, inplace=True)
        s_bi_count_in_7 = userSub.copy()
        # print(s_bi_count_in_7.info())
        # print(s_bi_count_in_7.head())

        s_bi_count_in_n = pd.merge(s_bi_count_in_7, s_bi_count_in_5, on=['shop_id'], how='left').fillna(0)
        s_bi_count_in_n = pd.merge(s_bi_count_in_n, s_bi_count_in_3, on=['shop_id'], how='left').fillna(0)
        s_bi_count_in_n = pd.merge(s_bi_count_in_n, s_bi_count_in_1, on=['shop_id'], how='left').fillna(0)
        # print(s_bi_count_in_n.info())
        # print(s_bi_count_in_n.head())
        # s_bi_count_in_n.to_csv('../data/features/features_1_4/s_bi_count_in_n.csv', index=False, encoding='utf-8')
    shop_features = pd.merge(shop_features, s_bi_count_in_n, on='shop_id', how='left').fillna(0)

    # (4)s_b_count_in_n(店铺在考察日前n天的行为总数计数   反映了店铺的热度（用户停留性）)
    Path = '../data/features/features_1_4/s_b_count_in_n.csv'
    if os.path.exists(Path):
        s_b_count_in_n = pd.read_csv(Path)
    else:
        userSub = s_bi_count_in_n.copy()
        userSub['s_b_count_in_3'] = userSub['s_b1_count_in_3'] + userSub['s_b2_count_in_3'] + userSub[
            's_b3_count_in_3'] + userSub['s_b4_count_in_3']
        userSub['s_b_count_in_5'] = userSub['s_b1_count_in_5'] + userSub['s_b2_count_in_5'] + userSub[
            's_b3_count_in_5'] + userSub['s_b4_count_in_5']
        userSub['s_b_count_in_7'] = userSub['s_b1_count_in_7'] + userSub['s_b2_count_in_7'] + userSub[
            's_b3_count_in_7'] + userSub['s_b4_count_in_7']
        userSub = userSub[['shop_id', 's_b_count_in_3', 's_b_count_in_5', 's_b_count_in_7']]
        # print(userSub.info())
        # print(userSub.head())
        s_b_count_in_n = userSub.copy()
        # s_b_count_in_n.to_csv('../data/features/features_1_4/s_b_count_in_n.csv', index=False, encoding='utf-8')
    shop_features = pd.merge(shop_features, s_b_count_in_n, on='shop_id', how='left').fillna(0)

    # (5)s_b2_diff_day(店铺的点击购买平均时差,反映了店铺的购买决策时间特点)
    Path = '../data/features/features_1_4/s_b2_diff_day.csv'
    if os.path.exists(Path):
        s_b2_diff_day = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub.drop_duplicates(inplace=True)
        userSub = userSub[['shop_id', 'type_1', 'type_2', 'action_time']]

        usertmp1 = userSub[userSub['type_1'] == 1]
        usertmp1 = usertmp1.drop(['type_2'], axis=1)
        usertmp1.rename(columns={'action_time': 'action_time_1'}, inplace=True)
        usertmp2 = userSub[userSub['type_2'] == 1]
        usertmp2 = usertmp2.drop(['type_1'], axis=1)
        usertmp2.rename(columns={'action_time': 'action_time_2'}, inplace=True)

        buy_conuts = usertmp2.groupby(['shop_id'])['type_2'].count().reset_index()
        buy_conuts.rename(columns={'type_2': 's_b_count'}, inplace=True)

        usertmp1 = usertmp1.sort_values(['shop_id', 'type_1', 'action_time_1'])
        usertmp2 = usertmp2.sort_values(['shop_id', 'type_2', 'action_time_2'])
        usertmp1 = usertmp1.drop_duplicates(['shop_id', 'type_1'], keep='first', inplace=False)
        usertmp2 = usertmp2.drop_duplicates(['shop_id', 'type_2'], keep='last', inplace=False)

        usertmp = pd.merge(usertmp1, usertmp2, on=['shop_id'], how='left')
        usertmp.dropna(axis=0, how='any', inplace=True)
        usertmp = pd.merge(usertmp, buy_conuts, on=['shop_id'], how='left')

        usertmp['s_b2_diff_all_day'] = (
                pd.to_datetime(usertmp['action_time_2']) - pd.to_datetime(usertmp['action_time_1'])).dt.days
        usertmp['s_b2_diff_day'] = usertmp['s_b2_diff_all_day'] / usertmp['s_b_count']
        usertmp.drop(['type_1', 'type_2', 'action_time_1', 'action_time_2'], axis=1, inplace=True)
        s_b2_diff_day = usertmp.copy()
        # s_b2_diff_day.to_csv('../data/features/features_1_4/s_b2_diff_day.csv', index=False, encoding='utf-8')
    shop_features = pd.merge(shop_features, s_b2_diff_day, on='shop_id', how='left').fillna(0)

    # (6)店铺评论
    Path = '../data/features/features_1_4/s_comments_counts.csv'
    if os.path.exists(Path):
        s_comments_counts = pd.read_csv(Path)
    else:
        user_comment = pd.read_csv('../data/processsed_data/jdata_comment.csv')
        user_comment = user_comment[(user_comment['dt'] >= start_time) & (user_comment['dt'] <= end_time)]
        user_comment = user_comment[['sku_id', 'comments', 'good_comments', 'bad_comments']]
        user_comment = user_comment.groupby(['sku_id'], as_index=False).sum()
        userSub = pd.merge(userAll, user_comment, how='left', on='sku_id').fillna(0)
        userSub = userSub[['shop_id', 'comments', 'good_comments', 'bad_comments']]
        userSub = userSub.groupby(['shop_id'], as_index=False).sum()
        userSub.rename(
            columns={'comments': 's_comments', 'good_comments': 's_good_comments', 'bad_comments': 's_bad_comments'},
            inplace=True)
        # userSub.to_csv('../data/features/features_1_4/s_comments_counts.csv', index=False, encoding='utf-8')
        s_comments_counts = userSub[['shop_id', 's_comments', 's_good_comments', 's_bad_comments']].copy()
    shop_features = pd.merge(shop_features, s_comments_counts, on='shop_id', how='left').fillna(0)

    # (7)店铺信息
    Path = '../data/features/features_1_4/shop_info.csv'
    if os.path.exists(Path):
        shop_info = pd.read_csv(Path)
    else:
        userSub = pd.read_csv('../data/processsed_data/jdata_shop.csv')
        userSub = userSub[['shop_id', 'fans_num', 'vip_num', 'shop_score', 'shop_reg_time']]
        df_shop_reg_time = pd.get_dummies(userSub.shop_reg_time, prefix='shop_reg_tm')
        userSub = pd.concat([userSub, df_shop_reg_time], axis=1)
        del df_shop_reg_time
        userSub.drop('shop_reg_time', axis=1, inplace=True)
        userSub['or_shop_score'] = userSub['shop_score'].apply(lambda x: 1 if x >= 0 else 0)
        shop_info = userSub.copy()
        # shop_info.to_csv('../data/features/features_1_4/shop_info.csv', index=False, encoding='utf-8')
    shop_features = pd.merge(shop_features, shop_info, on='shop_id', how='left').fillna(0)

    filePath = '../data/features/features_1_4/shop_features.csv'
    shop_features.to_csv(filePath, index=False, encoding='utf-8')
    print('shopInfo')
    return shop_features


def get_user_cate_features():
    ### -----------------------------
    ### 统计用户 - 品类之间的特征主要目的是分析给定用户及品类之间所形成的关系
    ###    -  点击，加入购物车，购买，收藏的总次数(完成)
    ###    -  点击总天数(完成)
    ###    -  购买商品种类数量，点击商品种类数量(完成)
    ### -----------------------------

    # 构建UC类别特征
    user_cate_features = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
    user_cate_features = user_cate_features[['user_id', 'cate']].drop_duplicates()
    # 1、uc_bi_last_day(用户-类别对各项行为上一次发生距考察日的时差 ,反映了user_id -item_category的活跃时间特征)
    Path = '../data/features/features_1_4/uc_bi_last_day.csv'
    if os.path.exists(Path):
        uc_bi_last_day = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4', 'action_time']]
        userSub1 = userSub[(userSub.type_1 == 1)][['user_id', 'cate', 'type_1', 'action_time']]
        userSub2 = userSub[(userSub.type_2 == 1)][['user_id', 'cate', 'type_2', 'action_time']]
        userSub3 = userSub[(userSub.type_3 == 1)][['user_id', 'cate', 'type_3', 'action_time']]

        userSub1 = userSub1.sort_values(['user_id', 'cate', 'action_time'])
        # print(userSub1.head(20))
        userSub1 = userSub1.drop_duplicates(['user_id', 'cate'], keep='last')
        userSub1['uc_b1_last_day'] = (
                pd.to_datetime(label_time_start) - pd.to_datetime(userSub1['action_time'])).dt.days
        userSub1.drop(['action_time', 'type_1'], axis=1, inplace=True)
        # print(userSub1.head(20))

        userSub2 = userSub2.sort_values(['user_id', 'cate', 'action_time'])
        # print(userSub1.head(20))
        userSub2 = userSub2.drop_duplicates(['user_id', 'cate'], keep='last')
        userSub2['uc_b2_last_day'] = (
                pd.to_datetime(label_time_start) - pd.to_datetime(userSub2['action_time'])).dt.days
        userSub2.drop(['action_time', 'type_2'], axis=1, inplace=True)
        # print(userSub2.head(20))

        userSub3 = userSub3.sort_values(['user_id', 'cate', 'action_time'])
        # print(userSub3.head(20))
        userSub3 = userSub3.drop_duplicates(['user_id', 'cate'], keep='last')
        userSub3['uc_b3_last_day'] = (
                pd.to_datetime(label_time_start) - pd.to_datetime(userSub3['action_time'])).dt.days
        userSub3.drop(['action_time', 'type_3'], axis=1, inplace=True)
        # print(userSub3.head(20))

        uc_bi_last_day = pd.merge(userSub1, userSub2, how='left', on=['user_id', 'cate']).fillna(60)
        uc_bi_last_day = pd.merge(uc_bi_last_day, userSub3, how='left', on=['user_id', 'cate']).fillna(60)
        # print(uc_bi_last_hours.info())
        # print(uc_bi_last_hours.head())
        # uc_bi_last_day.to_csv('../data/features/features_1_4/uc_bi_last_day.csv', index=False, encoding='utf-8')
    user_cate_features = pd.merge(user_cate_features, uc_bi_last_day, on=['user_id', 'cate'], how='left').fillna(60)
    print('uc_bi_last_day')

    # (2)uc_b2_diff_day（用户的点击购买平均时差，反映了用户的购买决策时间习惯）
    Path = '../data/features/features_1_4/uc_b2_diff_day.csv'
    if os.path.exists(Path):
        uc_b2_diff_day = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub = userSub[['action_time', 'user_id', 'cate', 'type_1', 'type_2']]
        userSub.drop_duplicates(inplace=True)
        userSub = userSub.sort_values(by=['user_id', 'cate', 'action_time'], axis=0, ascending=False)
        # print(userSub.head(20))
        usertmp1 = userSub[userSub['type_1'] == 1].drop(['type_2'], axis=1)
        usertmp1 = usertmp1.drop_duplicates(['user_id', 'cate', 'type_1'], keep='last', inplace=False)
        usertmp1.rename(columns={'action_time': 'action_time_1'}, inplace=True)
        usertmp2 = userSub[userSub['type_2'] == 1].drop(['type_1'], axis=1)
        usertmp2 = usertmp2.drop_duplicates(['user_id', 'cate', 'type_2'], keep='first', inplace=False)
        usertmp2.rename(columns={'action_time': 'action_time_2'}, inplace=True)
        # print(usertmp1.head())
        # print(usertmp2.head())
        usertmp = pd.merge(usertmp1, usertmp2, how='inner', on=['user_id', 'cate'])
        # print(usertmp.head(20))

        userSub = userSub[userSub['type_2'] == 1].drop(['type_1'], axis=1)
        userSub = userSub.groupby(['user_id', 'cate'])['action_time'].nunique().reset_index(name='uc_ac_nunique')
        userSub = pd.merge(usertmp, userSub, on=['user_id', 'cate'], how='left')

        userSub['uc_b2_diff_day'] = (
                pd.to_datetime(userSub['action_time_2']) - pd.to_datetime(userSub['action_time_1'])).dt.days
        userSub['uc_b2_diff_day'] = userSub['uc_b2_diff_day'] / userSub['uc_ac_nunique']
        uc_b2_diff_day = userSub[['user_id', 'cate', 'uc_ac_nunique', 'uc_b2_diff_day']]
        # uc_b2_diff_day.to_csv('../data/features/features_1_4/uc_b2_diff_day.csv', index=False, encoding='utf-8')
    user_cate_features = pd.merge(user_cate_features, uc_b2_diff_day, on=['user_id', 'cate'], how='left').fillna(60)
    print('uc_b2_diff_day')

    # 2、uc_b_count_in_n(用户-类别对在考察日前n天的行为总数计数  反映了user_id - item_category的活跃度)
    Path = '../data/features/features_1_4/uc_b_count_in_n.csv'
    if os.path.exists(Path):
        uc_b_count_in_n = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] > end_time_1) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        userSub['uc_b_count_in_1'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        uc_b_count_in_1 = userSub.copy()
        # print(uc_b_count_in_1.info())
        # print(uc_b_count_in_1.head())

        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        userSub['uc_b_count_in_3'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        uc_b_count_in_3 = userSub.copy()
        # print(uc_b_count_in_3.info())
        # print(uc_b_count_in_3.head())

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        userSub['uc_b_count_in_5'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        uc_b_count_in_5 = userSub.copy()
        # print(uc_b_count_in_5.info())
        # print(uc_b_count_in_5.head())

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        userSub['uc_b_count_in_7'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        uc_b_count_in_7 = userSub.copy()
        # print(uc_b_count_in_7.info())
        # print(uc_b_count_in_7.head())

        userSub = userAll[(userAll['action_time'] >= end_time_10) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        userSub['uc_b_count_in_10'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        uc_b_count_in_10 = userSub.copy()
        # print(uc_b_count_in_10.info())
        # print(uc_b_count_in_10.head())

        uc_b_count_in_n = pd.merge(uc_b_count_in_10, uc_b_count_in_7, how='left').fillna(0)
        uc_b_count_in_n = pd.merge(uc_b_count_in_n, uc_b_count_in_5, how='left').fillna(0)
        uc_b_count_in_n = pd.merge(uc_b_count_in_n, uc_b_count_in_3, how='left').fillna(0)
        uc_b_count_in_n = pd.merge(uc_b_count_in_n, uc_b_count_in_1, how='left').fillna(0)
        # print(uc_b_count_in_n.info())
        # print(uc_b_count_in_n.head())

        # uc_b_count_in_n.to_csv('../data/features/features_1_4/uc_b_count_in_n.csv', index=False, encoding='utf-8')
    user_cate_features = pd.merge(user_cate_features, uc_b_count_in_n, on=['user_id', 'cate'], how='left').fillna(0)
    print('uc_b_count_in_n')

    # 3、uc_bi_count_in_n(用户-类别对在考察日前n天的各项行为计数 反映了user_id -item_category的活跃度，
    # 反映了user_id -item_category的各项操作的活跃度，对应着user_id -item_category的购买习惯)
    Path = '../data/features/features_1_4/uc_bi_count_in_n.csv'
    if os.path.exists(Path):
        uc_bi_count_in_n = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] > end_time_1) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub.rename(
            columns={'type_1': 'uc_b1_count_in_1', 'type_2': 'uc_b2_count_in_1', 'type_3': 'uc_b3_count_in_1',
                     'type_4': 'uc_b4_count_in_1'}, inplace=True)
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.head())
        uc_bi_count_in_1 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub.rename(
            columns={'type_1': 'uc_b1_count_in_3', 'type_2': 'uc_b2_count_in_3', 'type_3': 'uc_b3_count_in_3',
                     'type_4': 'uc_b4_count_in_3'}, inplace=True)
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.head())
        uc_bi_count_in_3 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub.rename(
            columns={'type_1': 'uc_b1_count_in_5', 'type_2': 'uc_b2_count_in_5', 'type_3': 'uc_b3_count_in_5',
                     'type_4': 'uc_b4_count_in_5'}, inplace=True)
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.head())
        uc_bi_count_in_5 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub.rename(
            columns={'type_1': 'uc_b1_count_in_7', 'type_2': 'uc_b2_count_in_7', 'type_3': 'uc_b3_count_in_7',
                     'type_4': 'uc_b4_count_in_7'}, inplace=True)
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.head())
        uc_bi_count_in_7 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_10) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub.rename(
            columns={'type_1': 'uc_b1_count_in_10', 'type_2': 'uc_b2_count_in_10', 'type_3': 'uc_b3_count_in_10',
                     'type_4': 'uc_b4_count_in_10'}, inplace=True)
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.head())
        uc_bi_count_in_10 = userSub.copy()

        uc_bi_count_in_n = pd.merge(uc_bi_count_in_10, uc_bi_count_in_7, how='left').fillna(0)
        uc_bi_count_in_n = pd.merge(uc_bi_count_in_n, uc_bi_count_in_5, how='left').fillna(0)
        uc_bi_count_in_n = pd.merge(uc_bi_count_in_n, uc_bi_count_in_3, how='left').fillna(0)
        uc_bi_count_in_n = pd.merge(uc_bi_count_in_n, uc_bi_count_in_1, how='left').fillna(0)
        # print(uc_bi_count_in_n.info())
        # print(uc_bi_count_in_n.head())

        # uc_bi_count_in_n.to_csv('../data/features/features_1_4/uc_bi_count_in_n.csv', index=False, encoding='utf-8')
    user_cate_features = pd.merge(user_cate_features, uc_bi_count_in_n, on=['user_id', 'cate'], how='left').fillna(0)
    print('uc_bi_count_in_n')

    # 4、uc_b_count_rank_in_n_in_u(用户-类别对的行为在用户所有商品中的排序  反映了user_id对item_category的行为偏好)
    Path = '../data/features/features_1_4/uc_b_count_rank_in_n_in_u.csv'
    if os.path.exists(Path):
        uc_b_count_rank_in_n_in_u = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] > end_time_1) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        userSub['b_count'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        userSub['uc_b_count_rank_in_1_in_u'] = userSub['b_count'].groupby(userSub['user_id']).rank(ascending=0,
                                                                                                   method='dense')
        userSub.drop('b_count', axis=1, inplace=True)
        uc_b_count_rank_in_1_in_u = userSub.copy()
        # print(uc_b_count_rank_in_1_in_u.info())
        # print(uc_b_count_rank_in_1_in_u.head(20))

        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        userSub['b_count'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        userSub['uc_b_count_rank_in_3_in_u'] = userSub['b_count'].groupby(userSub['user_id']).rank(ascending=0,
                                                                                                   method='dense')
        userSub.drop('b_count', axis=1, inplace=True)
        uc_b_count_rank_in_3_in_u = userSub.copy()
        # print(uc_b_count_rank_in_3_in_u.info())
        # print(uc_b_count_rank_in_3_in_u.head(20))

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        userSub['b_count'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        userSub['uc_b_count_rank_in_5_in_u'] = userSub['b_count'].groupby(userSub['user_id']).rank(ascending=0,
                                                                                                   method='dense')
        userSub.drop('b_count', axis=1, inplace=True)
        uc_b_count_rank_in_5_in_u = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        userSub['b_count'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        userSub['uc_b_count_rank_in_7_in_u'] = userSub['b_count'].groupby(userSub['user_id']).rank(ascending=0,
                                                                                                   method='dense')
        userSub.drop('b_count', axis=1, inplace=True)
        uc_b_count_rank_in_7_in_u = userSub.copy()

        uc_b_count_rank_in_n_in_u = pd.merge(uc_b_count_rank_in_7_in_u, uc_b_count_rank_in_5_in_u, how='left',
                                             on=['user_id', 'cate']).fillna(1000)  # 没有的置为1000(没有竞争力的一个靠后排名)
        uc_b_count_rank_in_n_in_u = pd.merge(uc_b_count_rank_in_n_in_u, uc_b_count_rank_in_3_in_u, how='left',
                                             on=['user_id', 'cate']).fillna(1000)
        uc_b_count_rank_in_n_in_u = pd.merge(uc_b_count_rank_in_n_in_u, uc_b_count_rank_in_1_in_u, how='left',
                                             on=['user_id', 'cate']).fillna(1000)
        # uc_b_count_rank_in_n_in_u.to_csv('../data/features/features_1_4/uc_b_count_rank_in_n_in_u.csv', index=False, encoding='utf-8')
        # print(uc_b_count_rank_in_n_in_u.info())
        # print(uc_b_count_rank_in_n_in_u.head())

    user_cate_features = pd.merge(user_cate_features, uc_b_count_rank_in_n_in_u, on=['user_id', 'cate'],
                                  how='left').fillna(0)
    print('uc_b_count_rank_in_n_in_u')

    # 5、uc_b2_rate(用户-类别的点击购买转化率  反映了item_category的购买决策操作特点)
    Path = '../data/features/features_1_4/uc_b2_rate.csv'
    if os.path.exists(Path):
        uc_b2_rate = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub.drop_duplicates(inplace=True)
        userSub = userSub[['user_id', 'cate', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'cate'], as_index=False).sum()
        userSub['uc_type_1_ratio'] = np.log1p(userSub['type_2']) - np.log1p(userSub['type_1'])
        userSub['uc_type_3_ratio'] = np.log1p(userSub['type_2']) - np.log1p(userSub['type_3'])
        userSub['uc_type_2_ratio'] = np.log1p(userSub['type_4']) - np.log1p(userSub['type_2'])
        userSub['uc_b2_rate'] = userSub['type_2'] / (
                userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']).map(
            lambda x: x if x != 0 else x + 1)
        # print(userSub.head())
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        uc_b2_rate = userSub.copy()
        # print(uc_b2_rate.info())
        # print(uc_b2_rate.head())
        # uc_b2_rate.to_csv('../data/features/features_1_4/uc_b2_rate.csv', index=False, encoding='utf-8')
    user_cate_features = pd.merge(user_cate_features, uc_b2_rate, on=['user_id', 'cate'], how='left').fillna(0.0)
    print('c_b2_rate')

    # (6)uc_active_days（用户的点击购买行为习惯）
    Path = '../data/features/features_1_4/uc_active_days.csv'
    if os.path.exists(Path):
        uc_active_days = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
        userSub = userSub[["user_id", "cate", "action_time"]]
        ## 用户-品类对考察周前一段时间内做出行为的天数
        userSub = userSub.groupby(["user_id", "cate"])['action_time'].nunique().reset_index()
        userSub.rename(columns={'action_time': 'uc_active_days'}, inplace=True)
        uc_active_days = userSub.copy()
        # print(uc_active_days.info())
        # print(uc_active_days.head())
        # uc_active_days.to_csv('../data/features/features_1_4/uc_active_days.csv', index=False, encoding='utf-8')
    user_cate_features = pd.merge(user_cate_features, uc_active_days, on=['user_id', 'cate'], how='left').fillna(0)
    print('uc_active_days')
    filePath = '../data/features/features_1_4/user_cate_features.csv'
    user_cate_features.to_csv(filePath, index=False, encoding='utf-8')
    print('user_cate_info')
    return user_cate_features


def get_user_shop_features():
    ### -----------------------------
    ### 统计用户 - 商户之间的特征主要目的是分析给定用户及商户之间所形成的关系
    ###    -  us_b_count_in_n(用户-类别对在考察日前n天的行为总数计数  反映了user_id - shop_ip的活跃度)
    ###    -  us_bi_count_in_n(用户-类别对在考察日前n天的各项行为计数 反映了user_id -item_category的活跃度，反映了user_id -item_category的各项操作的活跃度，对应着user_id -item_category的购买习惯)
    ###    -  us_bi_last_day(用户-类别对各项行为上一次发生距考察日的时差   反映了user_id -item_category的活跃时间特征)
    ###    -  us_b_count_rank_in_n_in_u(用户 - 类别对的行为在用户所有商品中的排序反映了user_id对item_category的行为偏好)
    ###    -
    ###    -
    ### -----------------------------

    # 构建US类别特征
    user_shop_features = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
    user_shop_features = user_shop_features[['user_id', 'shop_id']].drop_duplicates()
    # 1、us_bi_last_day(用户-店铺对各项行为上一次发生距考察日的天差,反映了user_id -shop_id的活跃时间特征)
    Path = '../data/features/features_1_4/us_bi_last_day.csv'
    if os.path.exists(Path):
        us_bi_last_day = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'action_time']]
        userSub1 = userSub[(userSub.type_1 == 1)][['user_id', 'shop_id', 'type_1', 'action_time']]
        userSub2 = userSub[(userSub.type_2 == 1)][['user_id', 'shop_id', 'type_2', 'action_time']]
        userSub3 = userSub[(userSub.type_3 == 1)][['user_id', 'shop_id', 'type_3', 'action_time']]

        userSub1 = userSub1.sort_values(['user_id', 'shop_id', 'action_time'])
        # print(userSub1.head(20))
        userSub1 = userSub1.drop_duplicates(['user_id', 'shop_id'], keep='last')
        userSub1['us_b1_last_day'] = (
                pd.to_datetime(label_time_start) - pd.to_datetime(userSub1['action_time'])).dt.days
        userSub1.drop(['action_time', 'type_1'], axis=1, inplace=True)
        # print(userSub1.head(20))

        userSub2 = userSub2.sort_values(['user_id', 'shop_id', 'action_time'])
        # print(userSub1.head(20))
        userSub2 = userSub2.drop_duplicates(['user_id', 'shop_id'], keep='last')
        userSub2['us_b2_last_day'] = (
                pd.to_datetime(label_time_start) - pd.to_datetime(userSub2['action_time'])).dt.days
        userSub2.drop(['action_time', 'type_2'], axis=1, inplace=True)
        # print(userSub2.head(20))

        userSub3 = userSub3.sort_values(['user_id', 'shop_id', 'action_time'])
        # print(userSub5.head(20))
        userSub3 = userSub3.drop_duplicates(['user_id', 'shop_id'], keep='last')
        userSub3['us_b3_last_day'] = (
                pd.to_datetime(label_time_start) - pd.to_datetime(userSub3['action_time'])).dt.days
        userSub3.drop(['action_time', 'type_3'], axis=1, inplace=True)
        # print(userSub5.head(20))

        us_bi_last_day = pd.merge(userSub1, userSub2, how='left', on=['user_id', 'shop_id']).fillna(60)
        us_bi_last_day = pd.merge(us_bi_last_day, userSub3, how='left', on=['user_id', 'shop_id']).fillna(60)
        # print(us_bi_last_day.info())
        # print(us_bi_last_day.head())
        # us_bi_last_day.to_csv('../data/features/features_1_4/us_bi_last_day.csv', index=False, encoding='utf-8')
    user_shop_features = pd.merge(user_shop_features, us_bi_last_day, on=['user_id', 'shop_id'], how='left').fillna(0)
    print('us_bi_last_day')

    # 2、us_b2_diff_day（用户的点击购买平均时差，反映了用户的购买决策时间习惯）
    Path = '../data/features/features_1_4/us_b2_diff_day.csv'
    if os.path.exists(Path):
        us_b2_diff_day = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub = userSub[['action_time', 'user_id', 'shop_id', 'type_1', 'type_2']]
        userSub.drop_duplicates(inplace=True)
        userSub = userSub.sort_values(by=['user_id', 'shop_id', 'action_time'], axis=0, ascending=False)
        # print(userSub.head(20))

        usertmp1 = userSub[userSub['type_1'] == 1].drop(['type_2'], axis=1)
        usertmp1 = usertmp1.drop_duplicates(['user_id', 'shop_id', 'type_1'], keep='last', inplace=False)
        usertmp1.rename(columns={'action_time': 'action_time_1'}, inplace=True)
        usertmp2 = userSub[userSub['type_2'] == 1].drop(['type_1'], axis=1)
        usertmp2 = usertmp2.drop_duplicates(['user_id', 'shop_id', 'type_2'], keep='first', inplace=False)
        usertmp2.rename(columns={'action_time': 'action_time_2'}, inplace=True)
        # print(usertmp1.head())
        # print(usertmp2.head())
        usertmp = pd.merge(usertmp1, usertmp2, how='inner', on=['user_id', 'shop_id'])
        # print(usertmp.head(20))

        userSub = userSub[userSub['type_2'] == 1].drop(['type_1'], axis=1)
        userSub = userSub.groupby(['user_id', 'shop_id'])['action_time'].nunique().reset_index(name='us_ac_nunique')
        userSub = pd.merge(usertmp, userSub, on=['user_id', 'shop_id'], how='left')

        userSub['us_b2_diff_day'] = (
                pd.to_datetime(userSub['action_time_2']) - pd.to_datetime(userSub['action_time_1'])).dt.days
        userSub['us_b2_diff_day'] = userSub['us_b2_diff_day'] / userSub['us_ac_nunique']
        us_b2_diff_day = userSub[['user_id', 'shop_id', 'us_ac_nunique', 'us_b2_diff_day']]
        # us_b2_diff_day.to_csv('../data/features/features_1_4/us_b2_diff_day.csv', index=False, encoding='utf-8')
    user_shop_features = pd.merge(user_shop_features, us_b2_diff_day, on=['user_id', 'shop_id'], how='left').fillna(60)
    print('us_b2_diff_day')

    # 3、us_b_count_in_n(用户-类别对在考察日前n天的行为总数计数  反映了user_id - shop_ip的活跃度)
    Path = '../data/features/features_1_4/us_b_count_in_n.csv'
    if os.path.exists(Path):
        us_b_count_in_n = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] > end_time_1) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        userSub['us_b_count_in_1'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        us_b_count_in_1 = userSub.copy()
        # print(us_b_count_in_1.info())
        # print(us_b_count_in_1.head())

        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        userSub['us_b_count_in_3'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        us_b_count_in_3 = userSub.copy()
        # print(us_b_count_in_3.info())
        # print(us_b_count_in_3.head())

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        userSub['us_b_count_in_5'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        us_b_count_in_5 = userSub.copy()
        # print(us_b_count_in_5.info())
        # print(us_b_count_in_5.head())

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        userSub['us_b_count_in_7'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        us_b_count_in_7 = userSub.copy()
        # print(us_b_count_in_7.info())
        # print(us_b_count_in_7.head())

        userSub = userAll[(userAll['action_time'] >= end_time_10) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        userSub['us_b_count_in_10'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        us_b_count_in_10 = userSub.copy()
        # print(us_b_count_in_10.info())
        # print(us_b_count_in_10.head())

        us_b_count_in_n = pd.merge(us_b_count_in_10, us_b_count_in_7, how='left').fillna(0)
        us_b_count_in_n = pd.merge(us_b_count_in_n, us_b_count_in_5, how='left').fillna(0)
        us_b_count_in_n = pd.merge(us_b_count_in_n, us_b_count_in_3, how='left').fillna(0)
        us_b_count_in_n = pd.merge(us_b_count_in_n, us_b_count_in_1, how='left').fillna(0)
        # print(us_b_count_in_n.info())
        # print(us_b_count_in_n.head())
        # us_b_count_in_n.to_csv('../data/features/features_1_4/us_b_count_in_n.csv', index=False, encoding='utf-8')
    user_shop_features = pd.merge(user_shop_features, us_b_count_in_n, on=['user_id', 'shop_id'], how='left').fillna(0)
    print('us_b_count_in_n')

    # 4、us_bi_count_in_n(用户-类别对在考察日前n天的各项行为计数 反映了user_id -item_category的活跃度，
    # 反映了user_id -item_category的各项操作的活跃度，对应着user_id -item_category的购买习惯)
    Path = '../data/features/features_1_4/us_bi_count_in_n.csv'
    if os.path.exists(Path):
        us_bi_count_in_n = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] > end_time_1) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub.rename(
            columns={'type_1': 'us_b1_count_in_1', 'type_2': 'us_b2_count_in_1', 'type_3': 'us_b3_count_in_1',
                     'type_4': 'us_b4_count_in_1'}, inplace=True)
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.head())
        us_bi_count_in_1 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub.rename(
            columns={'type_1': 'us_b1_count_in_3', 'type_2': 'us_b2_count_in_3', 'type_3': 'us_b3_count_in_3',
                     'type_4': 'us_b4_count_in_3'}, inplace=True)
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.head())
        us_bi_count_in_3 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub.rename(
            columns={'type_1': 'us_b1_count_in_5', 'type_2': 'us_b2_count_in_5', 'type_3': 'us_b3_count_in_5',
                     'type_4': 'us_b4_count_in_5'}, inplace=True)
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.head())
        us_bi_count_in_5 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub.rename(
            columns={'type_1': 'us_b1_count_in_7', 'type_2': 'us_b2_count_in_7', 'type_3': 'us_b3_count_in_7',
                     'type_4': 'us_b4_count_in_7'}, inplace=True)
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.head())
        us_bi_count_in_7 = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_10) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub.rename(
            columns={'type_1': 'us_b1_count_in_10', 'type_2': 'us_b2_count_in_7', 'type_3': 'us_b3_count_in_10',
                     'type_4': 'us_b4_count_in_10'}, inplace=True)
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        # print(userSub.info())
        # print(userSub.head())
        us_bi_count_in_10 = userSub.copy()

        us_bi_count_in_n = pd.merge(us_bi_count_in_10, us_bi_count_in_7, how='left').fillna(0)
        us_bi_count_in_n = pd.merge(us_bi_count_in_n, us_bi_count_in_5, how='left').fillna(0)
        us_bi_count_in_n = pd.merge(us_bi_count_in_n, us_bi_count_in_3, how='left').fillna(0)
        us_bi_count_in_n = pd.merge(us_bi_count_in_n, us_bi_count_in_1, how='left').fillna(0)
        # print(us_bi_count_in_n.info())
        # print(us_bi_count_in_n.head())
        # us_bi_count_in_n.to_csv('../data/features/features_1_4/us_bi_count_in_n.csv', index=False, encoding='utf-8')
    user_shop_features = pd.merge(user_shop_features, us_bi_count_in_n, on=['user_id', 'shop_id'], how='left').fillna(0)
    print('us_bi_count_in_n')

    # 5、us_b_count_rank_in_n_in_u(用户-类别对的行为在用户所有商品中的排序  反映了user_id对item_category的行为偏好)
    Path = '../data/features/features_1_4/us_b_count_rank_in_n_in_u.csv'
    if os.path.exists(Path):
        us_b_count_rank_in_n_in_u = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] > end_time_1) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        userSub['b_count'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        userSub['us_b_count_rank_in_1_in_u'] = userSub['b_count'].groupby(userSub['user_id']).rank(ascending=0,
                                                                                                   method='dense')
        userSub.drop('b_count', axis=1, inplace=True)
        us_b_count_rank_in_1_in_u = userSub.copy()
        # print(us_b_count_rank_in_1_in_u.info())
        # print(us_b_count_rank_in_1_in_u.head(20))

        userSub = userAll[(userAll['action_time'] >= end_time_3) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        userSub['b_count'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        userSub['us_b_count_rank_in_3_in_u'] = userSub['b_count'].groupby(userSub['user_id']).rank(ascending=0,
                                                                                                   method='dense')
        userSub.drop('b_count', axis=1, inplace=True)
        us_b_count_rank_in_3_in_u = userSub.copy()
        # print(us_b_count_rank_in_3_in_u.info())
        # print(us_b_count_rank_in_3_in_u.head(20))

        userSub = userAll[(userAll['action_time'] >= end_time_5) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        userSub['b_count'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        userSub['us_b_count_rank_in_5_in_u'] = userSub['b_count'].groupby(userSub['user_id']).rank(ascending=0,
                                                                                                   method='dense')
        userSub.drop('b_count', axis=1, inplace=True)
        us_b_count_rank_in_5_in_u = userSub.copy()

        userSub = userAll[(userAll['action_time'] >= end_time_7) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        userSub['b_count'] = userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        userSub['us_b_count_rank_in_7_in_u'] = userSub['b_count'].groupby(userSub['user_id']).rank(ascending=0,
                                                                                                   method='dense')
        userSub.drop('b_count', axis=1, inplace=True)
        us_b_count_rank_in_7_in_u = userSub.copy()
        # 没有的置为1000(没有竞争力的一个靠后排名)
        us_b_count_rank_in_n_in_u = pd.merge(us_b_count_rank_in_7_in_u, us_b_count_rank_in_5_in_u, how='left',
                                             on=['user_id', 'shop_id']).fillna(1000)
        us_b_count_rank_in_n_in_u = pd.merge(us_b_count_rank_in_n_in_u, us_b_count_rank_in_3_in_u, how='left',
                                             on=['user_id', 'shop_id']).fillna(1000)
        us_b_count_rank_in_n_in_u = pd.merge(us_b_count_rank_in_n_in_u, us_b_count_rank_in_1_in_u, how='left',
                                             on=['user_id', 'shop_id']).fillna(1000)
        # us_b_count_rank_in_n_in_u.to_csv('../data/features/features_1_4/us_b_count_rank_in_n_in_u.csv', index=False, encoding='utf-8')
        # print(us_b_count_rank_in_n_in_u.info())
        # print(us_b_count_rank_in_n_in_u.head())
    user_shop_features = pd.merge(user_shop_features, us_b_count_rank_in_n_in_u, on=['user_id', 'shop_id'],
                                  how='left').fillna(0)

    # 6、us_b2_rate(用户-类别的点击购买转化率  反映了item_category的购买决策操作特点)
    Path = '../data/features/features_1_4/us_b2_rate.csv'
    if os.path.exists(Path):
        us_b2_rate = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub.drop_duplicates(inplace=True)
        userSub = userSub[['user_id', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['user_id', 'shop_id'], as_index=False).sum()
        userSub['us_type_1_ratio'] = np.log1p(userSub['type_2']) - np.log1p(userSub['type_1'])
        userSub['us_type_3_ratio'] = np.log1p(userSub['type_2']) - np.log1p(userSub['type_3'])
        userSub['us_type_2_ratio'] = np.log1p(userSub['type_4']) - np.log1p(userSub['type_2'])
        userSub['us_b2_rate'] = userSub['type_2'] / (
                userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']).map(
            lambda x: x if x != 0 else x + 1)
        # print(userSub.head())
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        us_b2_rate = userSub.copy()
        # print(c_b2_rate.info())
        # print(c_b2_rate.head())
        # us_b2_rate.to_csv('../data/features/features_1_4/us_b2_rate.csv', index=False, encoding='utf-8')
    user_shop_features = pd.merge(user_shop_features, us_b2_rate, on=['user_id', 'shop_id'], how='left').fillna(0.0)

    # 7、us_active_days（用户的点击购买行为习惯）
    Path = '../data/features/features_1_4/us_active_days.csv'
    if os.path.exists(Path):
        us_active_days = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
        userSub = userSub[["user_id", "shop_id", "action_time"]]
        # 用户-店铺对考察周前一段时间内做出行为的天数
        userSub = userSub.groupby(["user_id", "shop_id"])['action_time'].nunique().reset_index()
        userSub.rename(columns={'action_time': 'us_active_days'}, inplace=True)
        us_active_days = userSub.copy()
        # print(us_active_days.info())
        # print(us_active_days.head())
        # us_active_days.to_csv('../data/features/features_1_4/us_active_days.csv', index=False, encoding='utf-8')
    user_shop_features = pd.merge(user_shop_features, us_active_days, on=['user_id', 'shop_id'], how='left').fillna(0)

    filePath = '../data/features/features_1_4/user_shop_features.csv'
    user_shop_features.to_csv(filePath, index=False, encoding='utf-8')
    print('user_shop_info')
    return user_shop_features


def get_cate_shop_features():
    ### -----------------------------
    ### 统计品类 - 商户之间的特征主要目的是分析给定品类及商户之间所形成的关系
    ###    -  品类在所属商店中的销量排序
    ###    -  品类在所属商店中的销量排序
    ###    -  用户-品类-店铺对在考察日前n天的行为总数计数
    ###    -  用户-品类-店铺对在考察日前n天的各项行为计数
    ###    -  用户-品类-店铺对各项行为上一次发生距考察日的时差
    ###    -  用户-品类-店铺对的行为在用户所有商品中的排序
    ### -----------------------------

    userAll.drop_duplicates(inplace=True)
    # print(userAll.info())
    # 构建CS类别特征
    cate_shop_features = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
    cate_shop_features = cate_shop_features[['cate', 'shop_id']].drop_duplicates()
    # （1）cs_u_rank_in_c(品类在所属店铺中的用户人数排序，反映了cate在shop_id中的热度排名（用户覆盖性）)
    Path = '../data/features/features_1_4/cs_u_rank_in_s.csv'
    if os.path.exists(Path):
        cs_u_rank_in_s = pd.read_csv(Path)
    else:

        # 相同用户多次购买同一类产品,多次计数
        userSub = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['cate', 'shop_id']]
        # print(userSub.info())
        # print(userSub.head())
        userSub.insert(2, 'user_count', value=1)
        # userSub.loc[:, 'user_count'] = 1
        userSub = userSub.groupby(['cate', 'shop_id'], as_index=False).sum()
        userSub = userSub.sort_values(by=['cate', 'shop_id'])
        # print(userSub.info())
        # print(userSub.head())
        # print(userSub.user_count.max())
        userSub.loc[:, 'cs_u_rank_in_s'] = userSub['user_count'].groupby(userSub['shop_id']).rank(ascending=0,
                                                                                                  method='dense')
        userSub.drop('user_count', axis=1, inplace=True)
        # print(userSub.info())
        # print(userSub.head(30))
        cs_u_rank_in_s = userSub[['cate', 'shop_id', 'cs_u_rank_in_s']].copy()
        # cs_u_rank_in_s.to_csv('../data/features/features_1_4/cs_u_rank_in_s.csv', index=False, encoding='utf-8')
    cate_shop_features = pd.merge(cate_shop_features, cs_u_rank_in_s, on=['shop_id', 'cate'], how='left').fillna(0)
    print('cs_u_rank_in_s')

    # (2)cs_item_count(店铺品类下商品的数量)
    Path = '../data/features/features_1_4/cs_item_count.csv'
    if os.path.exists(Path):
        cs_item_count = pd.read_csv(Path)
    else:
        userSub = jdata_product[['sku_id', 'cate', 'shop_id']]
        userSub = userSub.groupby(['shop_id', 'cate'])['sku_id'].nunique().reset_index()
        userSub.rename(columns={'sku_id': 'cs_item_count'}, inplace=True)
        cs_item_count = userSub.copy()
        # cs_item_count.to_csv('../data/features/features_1_4/cs_item_count.csv', index=False, encoding='utf-8')
    cate_shop_features = pd.merge(cate_shop_features, cs_item_count, on=['shop_id', 'cate'], how='left').fillna(0)
    print('cs_item_count')

    # (3)cs_new_item_count(店铺品类下新品的数量)
    Path = '../data/features/features_1_4/cs_new_item_count.csv'
    if os.path.exists(Path):
        cs_new_item_count = pd.read_csv(Path)
    else:
        userSub = jdata_product[['shop_id', 'cate', 'market_tm']].copy()
        df_market_tm = pd.get_dummies(userSub.market_tm, prefix='cs_market_tm')
        userSub = pd.concat([userSub, df_market_tm], axis=1)
        del df_market_tm
        userSub.drop(['market_tm'], axis=1, inplace=True)
        userSub = userSub.groupby(['shop_id', 'cate'], as_index=False).sum()
        cs_new_item_count = userSub.copy()
        # cs_new_item_count.to_csv('../data/features/features_1_4/cs_new_item_count.csv', index=False, encoding='utf-8')
    cate_shop_features = pd.merge(cate_shop_features, cs_new_item_count, on=['shop_id', 'cate'], how='left').fillna(0)
    print('cs_new_item_count')

    # (4)cs_b2_count(类别被购买的频次)
    Path = '../data/features/features_1_4/cs_b2_count.csv'
    if os.path.exists(Path):
        cs_b2_count = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub = userSub[['shop_id', 'cate', 'type_2']].copy()
        userSub = userSub[(userSub.type_2 == 1)]
        cs_b2_count = userSub.groupby(['shop_id', 'cate'])['type_2'].count().reset_index(name='cs_b2_count')
        # cs_b2_count.to_csv('../data/features/features_1_4/cs_b2_count.csv', index=False, encoding='utf-8')
    cate_shop_features = pd.merge(cate_shop_features, cs_b2_count, on=['shop_id', 'cate'], how='left').fillna(0)
    print('cs_b2_count')

    # （2）cs_b_rank_in_s(品类在所属店铺中的行为总数排序 ，反映了cate在shop_id中的热度排名（用户停留性）)
    Path = '../data/features/features_1_4/cs_b_rank_in_s.csv'
    if os.path.exists(Path):
        cs_b_rank_in_s = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        # print(userSub.info())
        # print(userSub.head())
        userSub.insert(6, 'b_count',
                       value=userAll['type_1'] + userAll['type_2'] + userAll['type_3'] + userAll['type_4'])
        # print(userSub.info())
        # print(userSub.head())
        userSub = userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1)
        # print(userSub.head())
        userSub = userSub.groupby(['cate', 'shop_id'], as_index=False).sum()
        # print(userSub.head(50))
        # print(userSub.b_count.max())
        userSub['cs_b_rank_in_s'] = userSub['b_count'].groupby(userSub['shop_id']).rank(ascending=0, method='dense')
        # print(userSub.head(30))
        userSub.drop(['b_count'], axis=1, inplace=True)
        # userSub.to_csv('../data/features/features_1_4/cs_b_rank_in_s.csv', index=False, encoding='utf-8')
        cs_b_rank_in_s = userSub[['cate', 'shop_id', 'cs_b_rank_in_s']].copy()

    cate_shop_features = pd.merge(cate_shop_features, cs_b_rank_in_s, on=['cate', 'shop_id'], how='left').fillna(0)
    print('cs_b_rank_in_s')

    # （3)cs_b2_rank_in_s(品类在所属店铺中的销量排序 反映了cate在shop_id中的热度排名（销量）)
    Path = '../data/features/features_1_4/cs_b2_rank_in_s.csv'
    if os.path.exists(Path):
        cs_b2_rank_in_s = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
        userSub = userSub[['cate', 'shop_id', 'type_2']]
        userSub = userSub.groupby(['cate', 'shop_id'], as_index=False).sum()
        userSub['cs_b2_rank_in_s'] = userSub['type_2'].groupby(userSub['shop_id']).rank(ascending=0, method='dense')
        # print(userSub.info())
        # print(userSub.head())
        # print(userSub.type_2.max())
        userSub.drop('type_2', axis=1, inplace=True)
        # userSub.to_csv('../data/features/features_1_4/cs_b2_rank_in_s.csv', index=False, encoding='utf-8')
        cs_b2_rank_in_s = userSub[['cate', 'shop_id', 'cs_b2_rank_in_s']].copy()

    cate_shop_features = pd.merge(cate_shop_features, cs_b2_rank_in_s, on=['cate', 'shop_id'], how='left').fillna(0)

    # （4）cs_b2_rate(店铺下类别的点击购买转化率  反映了item_category的购买决策操作特点)
    Path = '../data/features/features_1_4/cs_b2_rate.csv'
    if os.path.exists(Path):
        cs_b2_rate = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] <= end_time)]
        userSub.drop_duplicates(inplace=True)
        userSub = userSub[['cate', 'shop_id', 'type_1', 'type_2', 'type_3', 'type_4']]
        userSub = userSub.groupby(['cate', 'shop_id'], as_index=False).sum()
        userSub['cs_type_1_ratio'] = np.log1p(userSub['type_2']) - np.log1p(userSub['type_1'])
        userSub['cs_type_3_ratio'] = np.log1p(userSub['type_2']) - np.log1p(userSub['type_3'])
        userSub['cs_type_2_ratio'] = np.log1p(userSub['type_4']) - np.log1p(userSub['type_2'])
        userSub['cs_b2_rate'] = userSub['type_2'] / (
                userSub['type_1'] + userSub['type_2'] + userSub['type_3'] + userSub['type_4']).map(
            lambda x: x if x != 0 else x + 1)
        # print(userSub.head())
        userSub.drop(['type_1', 'type_2', 'type_3', 'type_4'], axis=1, inplace=True)
        cs_b2_rate = userSub.copy()
        # print(cs_b2_rate.info())
        # print(cs_b2_rate.head())
        # cs_b2_rate.to_csv('../data/features/features_1_4/cs_b2_rate.csv', index=False, encoding='utf-8')
    cate_shop_features = pd.merge(cate_shop_features, cs_b2_rate, on=['cate', 'shop_id'], how='left').fillna(0.0)

    # (5)cs_active_days（用户的点击购买行为习惯）
    Path = '../data/features/features_1_4/cs_active_days.csv'
    if os.path.exists(Path):
        cs_active_days = pd.read_csv(Path)
    else:
        userSub = userAll[(userAll['action_time'] >= start_time) & (userAll['action_time'] <= end_time)]
        userSub = userSub[["cate", "shop_id", "action_time"]]
        # 品类-店铺对考察周前一段时间内被做出行为的天数
        userSub = userSub.groupby(["cate", "shop_id"])['action_time'].nunique().reset_index()
        userSub.rename(columns={'action_time': 'cs_active_days'}, inplace=True)
        cs_active_days = userSub.copy()
        # print(cs_active_days.info())
        # print(cs_active_days.head())
        # cs_active_days.to_csv('../data/features/features_1_4/cs_active_days.csv', index=False, encoding='utf-8')
    cate_shop_features = pd.merge(cate_shop_features, cs_active_days, on=['cate', 'shop_id'], how='left').fillna(0)

    # (6)是否是主营品类
    Path = '../data/features/features_1_4/cs_major_cate.csv'
    if os.path.exists(Path):
        cs_major_cate = pd.read_csv(Path)
    else:
        usertmp = userAll[['cate', 'shop_id']]
        usertmp.drop_duplicates(inplace=True)
        userSub = pd.read_csv('../data/processsed_data/jdata_shop.csv')
        userSub = userSub[['cate', 'shop_id']]
        userSub['cs_major_cate'] = userSub['cate'].apply(lambda x: 1 if x >= 1 else 0)
        userSub = userSub[(userSub.cs_major_cate == 1)]
        userSub = pd.merge(usertmp, userSub, how='left', on=['cate', 'shop_id']).fillna(0)
        cs_major_cate = userSub.copy()
        # cs_major_cate.to_csv('../data/features/features_1_4/cs_major_cate.csv', index=False, encoding='utf-8')
    cate_shop_features = pd.merge(cate_shop_features, cs_major_cate, on=['cate', 'shop_id'], how='left').fillna(0)

    filePath = '../data/features/features_1_4/cate_shop_features.csv'
    cate_shop_features.to_csv(filePath, index=False, encoding='utf-8')
    print('cate_shop_info')
    return cate_shop_features


if __name__ == '__main__':
    jdata_action = pd.read_csv('../data/processsed_data/jdata_action.csv')
    jdata_product = pd.read_csv('../data/processsed_data/jdata_product.csv')
    userAll = jdata_action.merge(jdata_product, on=['sku_id'])
    typeDummies = pd.get_dummies(userAll['type'], prefix='type')  # onehot哑变量编码
    userAll = pd.concat([userAll, typeDummies], axis=1)  # 将哑变量特征加入到表中
    # print(userSub.info())
    # print(userSub.head())
    userAll.drop('type', axis=1, inplace=True)

    # 训练集1时间变量
    start_time = '2018-03-12'
    end_time = '2018-03-25'
    end_time_1 = '2018-03-24'
    end_time_3 = '2018-03-23'
    end_time_5 = '2018-03-21'
    end_time_7 = '2018-03-19'
    end_time_10 = '2018-03-16'
    label_time_start = '2018-03-26'
    label_time_end = '2018-04-01'

    # # 训练集2时间变量
    # start_time  = '2018-03-19'
    # end_time    = '2018-04-01'
    # end_time_1  = '2018-03-31'
    # end_time_3  = '2018-03-30'
    # end_time_5  = '2018-03-28'
    # end_time_7  = '2018-03-26'
    # end_time_10 = '2018-03-23'
    # label_time_start = '2018-04-02'
    # label_time_end = '2018-04-08'

    # # 验证集时间变量
    # start_time  = '2018-03-26'
    # end_time    = '2018-04-08'
    # end_time_1  = '2018-04-07'
    # end_time_3  = '2018-04-06'
    # end_time_5  = '2018-04-04'
    # end_time_7  = '2018-04-02'
    # end_time_10 = '2018-03-30'
    # label_time_start = '2018-04-09'
    # label_time_end = '2018-04-15'

    # 测试集时间变量
    # start_time  = '2018-04-02'
    # end_time    = '2018-04-15'
    # end_time_1  = '2018-04-14'
    # end_time_3  = '2018-04-13'
    # end_time_5  = '2018-04-11'
    # end_time_7  = '2018-04-09'
    # end_time_10 = '2018-04-06'
    # label_time_start = '2018-04-16'
    # label_time_end   = '2018-04-22'
    #
    get_user_features()
    get_ucs_features()
    get_cateInfo_features()
    get_shopInfo_features()
    get_user_cate_features()
    get_user_shop_features()
    get_cate_shop_features()
