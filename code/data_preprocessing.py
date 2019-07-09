#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/05/18  10:20
# @Author  : chensw、wangwei
# @File    : data_preprocessing.py
# @Describe: 标明文件实现的功能
# @Modify  : 修改的地方

import pandas as pd


def user_reg_tm(times):
    if times < '2008-01-01':
        return 4
    elif times < '2015-01-01' and times >= '2008-01-01':
        return 3
    elif times < '2018-01-01' and times >= '2015-01-01':
        return 2
    elif times >= '2018-01-01':
        return 1


def shop_reg_tm(times):
    if times < '2013-01-01' and times >= '2010-01-01':
        return 5
    elif times < '2015-01-01' and times >= '2013-01-01':
        return 4
    elif times < '2017-01-01' and times >= '2015-01-01':
        return 3
    elif times < '2018-01-01' and times >= '2017-01-01':
        return 2
    elif times >= '2018-01-01':
        return 1


def item_market_tm(times):
    if times < '2012-01-01':
        return 6
    elif times < '2015-01-01' and times >= '2012-01-01':
        return 5
    elif times < '2016-01-01' and times >= '2015-01-01':
        return 4
    elif times < '2017-01-01' and times >= '2016-01-01':
        return 3
    elif times < '2018-01-01' and times >= '2017-01-01':
        return 2
    elif times >= '2018-01-01':
        return 1


if __name__ == '__main__':
    # 读取原始数据
    actionInfo = pd.read_csv('../data/original_data/jdata_action.csv')
    userInfo = pd.read_csv('../data/original_data/jdata_user.csv')
    shopInfo = pd.read_csv('../data/original_data/jdata_shop.csv')
    itemInfo = pd.read_csv('../data/original_data/jdata_product.csv')
    commentInfo = pd.read_csv('../data/original_data/jdata_comment.csv')

    # 对行为数据进行初步预处理
    # 用户行为时间只取天数
    actionInfo['action_time'] = pd.to_datetime(actionInfo.action_time.values).date
    actionInfo.to_csv('../data/processsed_data/jdata_action.csv', index=False, encoding='utf-8')

    # 对物品评分数据进行初步预处理
    commentInfo['dt'] = commentInfo['dt'].apply(lambda x: str(x))
    commentInfo.to_csv('../data/processsed_data/jdata_comment.csv', index=False, encoding='utf-8')

    # 对用户数据进行初步预处理
    # 用户注册时间只取天数
    userInfo['user_reg_tm'] = pd.to_datetime(userInfo.user_reg_tm.values).date
    userInfo.user_reg_tm.fillna(userInfo.user_reg_tm.mode()[0], inplace=True)
    userInfo['user_reg_tm'] = userInfo['user_reg_tm'].apply(lambda x: str(x))
    userInfo['user_reg_time'] = userInfo['user_reg_tm'].apply(lambda x: user_reg_tm(x))
    print('Check any missing value?\n', userInfo.isnull().any())
    userInfo.to_csv('../data/processsed_data/jdata_user.csv', index=False, encoding='utf-8')

    # 对商家店铺数据进行初步预处理
    # 开店时间只取天数
    shopInfo['shop_reg_tm'] = pd.to_datetime(shopInfo.shop_reg_tm.values).date
    shopInfo.shop_reg_tm.fillna(shopInfo.shop_reg_tm.mode()[0], inplace=True)
    shopInfo['shop_reg_tm'] = shopInfo['shop_reg_tm'].apply(lambda x: str(x))
    shopInfo['shop_reg_time'] = shopInfo['shop_reg_tm'].apply(lambda x: shop_reg_tm(x))
    print('Check any missing value?\n', shopInfo.isnull().any())
    shopInfo.to_csv('../data/processsed_data/jdata_shop.csv', index=False, encoding='utf-8')
    print(shopInfo.info())

    # 对商品数据进行初步预处理
    # 商品上市时间只取天数
    itemInfo['market_time'] = pd.to_datetime(itemInfo.market_time.values).date
    itemInfo.market_time.fillna(itemInfo.market_time.mode()[0], inplace=True)
    itemInfo['market_time'] = itemInfo['market_time'].apply(lambda x: str(x))
    itemInfo['market_tm'] = itemInfo['market_time'].apply(lambda x: item_market_tm(x))
    print('Check any missing value?\n', itemInfo.isnull().any())
    itemInfo.to_csv('../data/processsed_data/jdata_product.csv', index=False, encoding='utf-8')
    print(itemInfo.info())





