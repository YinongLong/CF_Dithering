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


def evaluation():
    """
    对推荐的效果进行探索
    :return: 
    """
    rating_data = utility.load_train_data()
    test_data = utility.load_test_data()
    test_data = process_test_data(test_data)
    movie_info = utility.load_movie_info()

    ucf = ItemCF()
    ucf.train(rating_data)
    user_id = 1
    k = 10
    prediction = ucf.predict_single(user_id, k)
    goals = sorted(test_data[user_id].items(), key=lambda item: item[1], reverse=True)
    goals = [item_score[0] for item_score in goals]

    count = 0
    hit = 0
    hit_movies = []
    for item_id in prediction:
        count += 1
        if item_id in goals:
            hit += 1
            hit_movies.append(item_id)
    print('The precision is %.2f' % (hit / count))

    print('==' * 40)
    print('The %d prediction result is:' % k)
    for item_id in prediction:
        print(movie_info[item_id], end=' | ')
    print()
    print('==' * 40)
    print('The user has rated is:')
    for item_id in goals:
        if item_id in hit_movies:
            prefix = '**' * 4
        else:
            prefix = ''
        print(prefix, movie_info[item_id], end=' | ')
    print()
    print('==' * 40)
    print('And the hit movies are:')
    for item_id in hit_movies:
        print(movie_info[item_id], end=' | ')


def main():
    evaluation()


if __name__ == '__main__':
    main()