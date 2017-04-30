# -*- coding: utf-8 -*-
"""
Created on 2017/4/18 15:01

@author: YinongLong
@file: collaborative_filtering.py

关于用户和物品的相似度计算采用0/1评分方式

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
        处理用户物品评分数据，分别生成以用户、物品为关键字的字典，方便后续的计算。
        
        :param rating_data: list，用户评分记录的三元组
        :return: None
        """
        for user_id, item_id, score in rating_data:
            self.user_data[user_id][item_id] = np.float32(score)
            self.item_data[item_id][user_id] = np.float32(score)

    def _similarity_user(self, user_1, user_2):
        """
        计算两个用户之间的cosine相似度，同时添加对热门物品的惩罚
        
        :param user_1: int，用户ID
        :param user_2: int，用户ID
        :return: float，两个用户之间的cosine相似度
        """
        records_1 = self.user_data[user_1]
        records_2 = self.user_data[user_2]
        numerator = 0.
        norm_user_1 = 0.
        for item_id, score in records_1.items():
            norm_user_1 += 1
            if item_id in records_2:
                nums_popularity = len(self.item_data[item_id])
                numerator += 1.0 / nums_popularity
        norm_user_2 = 0.
        for _, score in records_2.items():
            norm_user_2 += 1
        denominator = np.sqrt(norm_user_1) * np.sqrt(norm_user_2)
        return numerator / denominator

    def dithering_predict(self, user_id, alpha=None, beta=None, k=10, num_dithering=10, num_users=5, seed=1, **kargs):
        """
        根据指定的用户ID，计算与其最为相似的num_dithering个用户，首先对这些相似的用户根据参数alpha和beta进行抖动，
        然后使用num_users用户的历史记录进行推荐计算。
        
        :param user_id: int，指定的用户ID
        :param alpha: float，抖动的参数，如果为None，代表不进行抖动
        :param beta: float，抖动的参数
        :param k: int，给用户推荐的物品数
        :param num_dithering: int，参与抖动的用户个数
        :param num_users: int，用来进行推荐计算的用户个数
        :param seed: int，设置随机数生成器的种子，使得实验可以复现
        :param kargs: dict，收集额外的关键字参数
        :return: 返回给用户推荐的k个物品ID，以及物品ID在未抖动结果中的rank
        """
        candidates = list()
        for uid in self.user_data.keys():
            if uid != user_id:
                candidates.append((uid, self._similarity_user(user_id, uid)))

        # 选择出最为相似的num_dithering个用户
        candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:num_dithering]
        used_items = self.user_data[user_id]
        # 计算抖动范围内的原始推荐结果
        original = collections.defaultdict(np.float32)
        for uid, similarity in candidates:
            records = self.user_data[uid]
            for item_id, score in records.items():
                if item_id not in used_items:
                    original[item_id] += score * similarity
        original = sorted(original.items(), key=lambda item: item[1], reverse=True)
        original_order = {item[0]: index for index, item in enumerate(original, start=1)}
        # 对推荐候选用户进行抖动
        if (alpha is not None) and (beta is not None):
            scores = list()
            np.random.seed(seed)
            for i in range(num_dithering):
                uid, _ = candidates[i]
                score = alpha * np.log(i + 1.0) + (1 - alpha) * np.random.normal(0., np.sqrt(np.sqrt(np.log(beta))))
                scores.append((uid, score))
            candidates = dict(candidates)
            # 根据抖动的结果进行排序
            scores.sort(key=lambda item: item[1], reverse=True)
            temp_candidates = [(uid, candidates[uid]) for uid, _ in scores]
            candidates = temp_candidates

        result = collections.defaultdict(np.float32)
        # 选择出用来计算推荐的用户
        candidates = candidates[:num_users]
        for uid, similarity in candidates:
            records = self.user_data[uid]
            for item_id, score in records.items():
                if item_id not in used_items:
                    result[item_id] += similarity * score
        result = sorted(result.items(), key=lambda item: item[1], reverse=True)
        return [result[i][0] for i in range(k)], [original_order[result[i][0]] for i in range(k)]

    def predict_single(self, user_id, k=10, num_users=5):
        """
        对指定的单个用户ID，计算最有可能的k个推荐物品ID
        
        :param user_id: int，待推荐的用户ID
        :param k: int，推荐的物品个数
        :param num_users: int，计算相似邻居用户的个数
        :return: list，推荐的物品ID列表
        """
        candidates = list()
        for uid in self.user_data.keys():
            if uid != user_id:
                candidates.append((uid, self._similarity_user(user_id, uid)))

        # 选择出最相似的num_users个用户
        candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:num_users]

        result = collections.defaultdict(np.float32)
        used_items = self.user_data[user_id]
        for uid, similarity in candidates:
            records = self.user_data[uid]
            for item_id, score in records.items():
                # 过滤掉用户已经使用过的物品
                if item_id not in used_items:
                    result[item_id] += similarity * score
        result = sorted(result.items(), key=lambda item: item[1], reverse=True)
        return [result[i][0] for i in range(k)]

    def predict(self, user_ids, k=10, num_users=5):
        """
        对指定的用户ID序列计算对每个用户的推荐物品ID
        
        :param user_ids: list，待推荐物品的用户ID列表 
        :param k: int，推荐物品的个数
        :param num_users: int，计算临近用户的个数
        :return: dict，以每个待推荐用户的ID为key，value为推荐结果
        """
        result = dict()
        for user_id in user_ids:
            result[user_id] = self.predict_single(user_id, k, num_users)
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
        处理用户物品评分数据三元组
        
        :param rating_data: list，用户、物品，以及评分三元组 
        :return: None
        """
        for user_id, item_id, score in rating_data:
            self.user_data[user_id][item_id] = np.float32(score)
            self.item_data[item_id][user_id] = np.float32(score)

    def _similarity_item(self, item_1, item_2):
        """
        计算两个指定物品ID的cosine相似度，添加了对活跃用户的惩罚
        
        :param item_1: int，物品ID
        :param item_2: int，物品ID
        :return: float，计算的两个物品的cosine相似度
        """
        records_1 = self.item_data[item_1]
        records_2 = self.item_data[item_2]
        numerator = 0.
        norm_item_1 = 0.
        for user, score in records_1.items():
            norm_item_1 += 1
            if user in records_2:
                num_popularity = len(self.user_data[user])
                numerator += 1.0 / num_popularity
        norm_item_2 = 0.
        for _, score in records_2.items():
            norm_item_2 += 1
        denominator = np.sqrt(norm_item_1) * np.sqrt(norm_item_2)
        return numerator / denominator

    def dithering_predict(self, user_id, alpha=None, beta=None, k=10, num_dithering=10, seed=1, **kargs):
        """
        根据指定的用户ID，给用户推荐k个物品，其中推荐物品来自对num_dithering个物品抖动后的结果，因此
        指定的参数中必须满足num_dithering >= k
        
        :param user_id: int，指定的用户ID
        :param alpha: float，抖动操作的参数，如果为None，则代表不进行抖动操作
        :param beta: float，抖动操作的参数，如果为None，则代表不进行抖动操作
        :param k: int，推荐的物品个数
        :param num_dithering: int，抖动的物品个数
        :param seed: int，随机数生成器的种子，使得实验可以复现
        :param kargs: dict，收集额外的关键字参数
        :return: 返回推荐的k个物品的ID，以及推荐结果在原始未抖动结果的rank
        """
        used_items = self.user_data[user_id]
        candidates = collections.defaultdict(np.float32)
        # 计算所有指定用户为评分的物品与用户已经评分的物品的相似度
        for iid in self.item_data.keys():
            if iid not in used_items:
                for item_id, score in used_items.items():
                    candidates[iid] += self._similarity_item(iid, item_id) * score
        candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:num_dithering]
        original_order = {item[0]: index for index, item in enumerate(candidates, start=1)}
        if (alpha is not None) and (beta is not None):
            scores = list()
            for i in range(num_dithering):
                iid, _ = candidates[i]
                score = alpha * np.log(i + 1.0) + (1 - alpha) * np.random.normal(0., np.sqrt(np.sqrt(np.log(beta))))
                scores.append((iid, score))
            scores.sort(key=lambda item: item[1], reverse=True)
            candidates = dict(candidates)
            temp_candidates = [(iid, candidates[iid]) for iid, _ in scores]
            candidates = temp_candidates[:k]
        return [candidates[i][0] for i in range(k)], [original_order[candidates[i][0]] for i in range(k)]

    def predict_single(self, user_id, k=10):
        """
        根据指定的用户ID，给用户推荐k个物品
        
        :param user_id: int，指定的用户ID
        :param k: int，推荐物品的个数
        :return: list，推荐的k个物品的ID，按照推荐强度从大到小排序
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
        对于给定的多个用户ID列表，对每一个用户推荐k个物品
        
        :param user_ids: list，多个用户ID的列表 
        :param k: int，推荐的物品个数
        :return: dict，以用户ID为key，推荐结果为value
        """
        result = dict()
        for user_id in user_ids:
            temp_result = self.predict_single(user_id, k)
            result[user_id] = temp_result
        return result


def process_test_data(rating_data):
    """
    处理用来评估推荐结果的测试数据，将其转化为以用户ID为key的字典
    
    :param rating_data: list，用户、物品，以及评分记录三元组 
    :return: dict，整理后的数据
    """
    dict_data = collections.defaultdict(dict)
    for user_id, item_id, score in rating_data:
        dict_data[user_id][item_id] = score
    return dict_data


def metric_precision(prediction, goals):
    """
    计算推荐结果的准确率，即所有推荐结果中，包含在用户已经评分的物品的比例
    
    :param prediction: list，预测的推荐列表
    :param goals: list，用户实际上已经评分的物品
    :return: 返回推荐结果的precision，以及命中的推荐物品ID
    """
    count = 0
    hit = 0
    hit_movies = list()
    for item_id in prediction:
        count += 1
        if item_id in goals:
            hit += 1
            hit_movies.append(item_id)
    return hit / count, hit_movies


def evaluate_dithering(user_id, method, movie_info, goals, k=10):
    """
    对Dithering的效果进行评估
    
    :param user_id: int，待推荐物品的用户ID
    :param method: object，待评估的推荐方法，可调用其dithering_predict方法
    :param movie_info: dict，电影ID对用的电影名称
    :param goals: list，用户现实中评分的物品，即预测推荐的目标
    :param k: int，给用户推荐的物品数
    :return: None
    """
    alphas = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9]
    betas = [1.1, 2.0, 5.0, 1.1, 2.0, 5.0, 1.1, 20.0, 5.0]

    result = dict()
    # 推荐不进行抖动的结果
    original, original_rank = method.dithering_predict(user_id, alpha=None, beta=None)
    # 推荐利用抖动的结果
    for alpha, beta in zip(alphas, betas):
        recommendations, rank = method.dithering_predict(user_id, alpha=alpha, beta=beta, seed=1)
        result[(alpha, beta)] = (recommendations, rank)

    precision, hit_movies = metric_precision(original, goals)
    print('对用户%d的原始推荐结果precision=%f，具体推荐为：' % (user_id, precision))
    for iid in original:
        print(movie_info[iid], end=' | ')
    print('\n\n')
    print('原始推荐结果的rank: ')
    for index in original_rank:
        print(index, end=' ')
    print('\n\n')

    print('推荐命中结果为：')
    for iid in hit_movies:
        print(movie_info[iid], end=' | ')
    print('\n\n')

    print('==' * 40)
    print('具体的各种抖动后的结果为：')
    for key, value in result.items():
        alpha, beta = key
        prediction, rank = value
        precision, hit_movies = metric_precision(prediction, goals)
        print('alpha=%f, beta=%f, precision=%f' % (alpha, beta, precision))
        print('推荐结果的rank: ')
        for index in rank:
            print(index, end=' ')
        print('\n\n')
        print('==' * 40)


def evaluation():
    """
    对推荐的效果进行测试
    
    :return: None
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
    np.random.seed(1)
    user_id = np.random.randint(1, 944)
    print('待推荐的用户ID:', user_id)
    k = 10

    # 获取测试数据中，指定用户已经评分的物品
    goals = sorted(test_data[user_id].items(), key=lambda item: item[1], reverse=True)
    goals = [item_score[0] for item_score in goals]

    evaluate_dithering(user_id, ucf, movie_info, goals, k=10)


def main():
    evaluation()


if __name__ == '__main__':
    main()