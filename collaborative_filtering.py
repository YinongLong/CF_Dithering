# -*- coding: utf-8 -*-
"""
Created on 2017/4/18 15:01

@author: YinongLong
@file: collaborative_filtering.py

"""
from __future__ import print_function
from __future__ import division

import collections

import numpy as np
import utility


class UserCF(object):
    """
    实现基于用户的协同过滤
    """
    def __init__(self):
        self.user_data = collections.defaultdict(dict)
        self.item_data = collections.defaultdict(dict)

    def train(self, rating_data):
        """
        处理用户-物品评分数据
        :param rating_data: 
        :return: 
        """
        for user_id, item_id, score in rating_data:
            self.user_data[user_id][item_id] = np.float32(score)
            self.item_data[item_id][user_id] = np.float32(score)

    def _similarity_user(self, user_1, user_2):
        """
        计算两个用户之间的相似度
        :param user_1: 
        :param user_2: 
        :return: 
        """
        records_1 = self.user_data[user_1]
        records_2 = self.user_data[user_2]
        numerator = 0.
        norm_user_1 = 0.
        for item_id, score in records_1.items():
            norm_user_1 += score * score
            if item_id in records_2:
                numerator += score * records_2[item_id]
        norm_user_2 = 0.
        for _, score in records_2.items():
            norm_user_2 += score * score
        denominator = np.sqrt(norm_user_1) * np.sqrt(norm_user_2)
        return numerator / denominator

    def predict_single(self, user_id, k=10):
        """
        对指定的单个用户ID，计算最有可能的k个推荐
        :param user_id: 
        :param k: 
        :return: 
        """
        candidates = list()
        for cid in self.user_data.keys():
            if cid != user_id:
                candidates.append((cid, self._similarity_user(user_id, cid)))
        result = collections.defaultdict(np.float32)
        for cid, similarity in candidates:
            records = self.user_data[cid]
            for item_id, score in records.items():
                result[item_id] += similarity * score
        result = sorted(result.items(), key=lambda item: item[1], reverse=True)
        return [result[i][0] for i in range(k)]

    def predict(self, user_ids, k=10):
        result = dict()
        for user_id in user_ids:
            result[user_id] = self.predict_single(user_id, k)
        return result


class ItemCF(object):
    """
    实现基于物品的协同过滤
    """
    def __init__(self):
        self.user_data = collections.defaultdict(dict)
        self.item_data = collections.defaultdict(dict)

    def train(self, rating_data):
        """
        处理用户-物品评分数据
        :param rating_data: 
        :return: 
        """
        for user_id, item_id, score in rating_data:
            self.user_data[user_id][item_id] = np.float32(score)
            self.item_data[item_id][user_id] = np.float32(score)

    def _similarity_item(self, item_1, item_2):
        """
        计算两个指定物品ID的相似度
        :param item_1: 
        :param item_2: 
        :return: 
        """
        records_1 = self.item_data[item_1]
        records_2 = self.item_data[item_2]
        numerator = 0.
        norm_item_1 = 0.
        for user, score in records_1.items():
            norm_item_1 += score * score
            if user in records_2:
                numerator += score * records_2[user]
        norm_item_2 = 0.
        for _, score in records_2.items():
            norm_item_2 += score * score
        denominator = np.sqrt(norm_item_1) * np.sqrt(norm_item_2)
        return numerator / denominator

    def predict_single(self, user_id, k=10):
        """
        根据指定的用户ID，给用户推荐top-K个电影
        :param user_id: int, 指定的用户ID
        :param k: int, 个数
        :return: list, Top-K个电影的ID，按照推荐强度从大到小排序
        """
        items = self.user_data[user_id]
        candidates = []
        for item_id, score in items.items():
            for c_item_id in self.item_data.keys():
                if c_item_id not in items:
                    candidates.append((c_item_id, score * self._similarity_item(item_id, c_item_id)))
        result = collections.defaultdict(np.float32)
        for item_id, priority in candidates:
            result[item_id] += priority
        result = sorted(result.items(), key=lambda item: item[1], reverse=True)
        return [result[i][0] for i in range(k)]

    def predict(self, user_ids, k=10):
        """
        预测给定用户ID列表中，用户最有可能观看的K部电影
        :param user_ids: 
        :param k:
        :return: 
        """
        result = dict()
        for user_id in user_ids:
            temp_result = self.predict_single(user_id, k)
            result[user_id] = temp_result
        return result


def process_test_data(rating_data):
    dict_data = collections.defaultdict(dict)
    for user_id, item_id, score in rating_data:
        dict_data[user_id][item_id] = score
    return dict_data


def dithering(recommendation, alpha, beta, seed=None):
    """
    对推荐列表recommendation进行抖动
    :param recommendation: list，根据推荐的强度从大到小排序的电影推荐列表
    :param alpha: float，抖动的alpha参数，其范围在0-1
    :param beta: float，抖动的beta参数
    :param seed: int，方便实验的重现
    :return: list，返回抖动后的推荐列表
    """
    len_rec = len(recommendation)
    result = list()
    if seed is not None:
        np.random.seed(seed)
    for i in range(1, len_rec + 1):
        item_id = recommendation[i-1]
        score = alpha * np.log(i) + (1 - alpha) * np.random.normal(loc=0.0, scale=np.log(beta))
        result.append((item_id, score))
    result.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in result]


def metric_precision(prediction, goals, movie_info):
    """
    计算推荐结果的准确率，即所有推荐结果中，包含在用户已经评分的物品的比例
    :param prediction: list，预测的推荐列表
    :param goals: list，用户实际上已经评分的物品
    :return: float，推荐结果中，命中用户兴趣的比例大小
    """
    count = 0
    hit = 0
    hit_movies = list()
    print('The prediction results are:')
    prediction_order = {}
    for item_id in prediction:
        print(movie_info[item_id], end=' | ')
        count += 1
        prediction_order[item_id] = count
        if item_id in goals:
            hit += 1
            hit_movies.append(item_id)
    print()
    return hit / count, hit_movies, prediction_order


def print_prediction_result(result, hit_movies, movies_info):
    print('==' * 40)
    print('The prediction precision is %.2f' % result)
    print()
    print('==' * 40)
    print('And the hit movies are :')
    for item_id in hit_movies:
        print(movies_info[item_id], end=' | ')
    print(end='\n\n')
    print('==' * 40)


def test_dithering(prediction, movie_info, prediction_order):
    """
    对系统的推荐进行抖动，观察效果
    :param prediction: list，推荐算法直接预测的结果
    :return: None
    """
    alphas = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9]
    betas = [1.1, 2.0, 5.0, 1.1, 2.0, 5.0, 1.1, 20.0, 5.0]
    result = dict()
    seed = 1
    for alpha, beta in zip(alphas, betas):
        temp_result = dithering(prediction, alpha, beta, seed)
        result[(alpha, beta)] = temp_result
    print('The original order of the prediction result:')
    for item_id in prediction:
        print(item_id, end=' ')
    print(end='\n\n\n')
    for key, values in result.items():
        print('--' * 40)
        print('Dithering of alpha=%f and beta=%f' % (key[0], key[1]), end='\n\n')
        for item_id in values:
            print(item_id, end=' ')
        print()
        for item_id in values:
            print(movie_info[item_id], end=' ')
        print()
        for item_id in values:
            print(prediction_order[item_id], end=' ')
        print(end='\n\n\n')


def evaluation():
    """
    对推荐的效果进行测试
    :return: 
    """
    # 读取已有的用户物品评分记录
    rating_data = utility.load_train_data()
    # 读取用来对推荐结果进行测试的记录数据
    test_data = utility.load_test_data()
    # 将三元组（用户，物品，评分）的测试数据处理成以用户为key，用户对物品的评分记录为value的字典
    test_data = process_test_data(test_data)
    # 读取电影ID对应的具体电影名称的数据
    movie_info = utility.load_movie_info()

    # 测试基于物品的协同过滤算法
    ucf = UserCF()
    ucf.train(rating_data)
    # 指定预测推荐的用户ID，以及推荐列表的长度
    user_id = 1
    k = 10
    # 给指定的用户进行推荐
    prediction = ucf.predict_single(user_id, k)

    # 获取测试数据中，指定用户已经评分的物品
    goals = sorted(test_data[user_id].items(), key=lambda item: item[1], reverse=True)
    goals = [item_score[0] for item_score in goals]

    precision, hit_movies, prediction_order = metric_precision(prediction, goals, movie_info)
    print_prediction_result(precision, hit_movies, movie_info)

    test_dithering(prediction, movie_info, prediction_order)


def main():
    evaluation()


if __name__ == '__main__':
    main()