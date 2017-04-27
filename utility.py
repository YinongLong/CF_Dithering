# -*- coding: utf-8 -*-
"""
Created on 2017/4/18 15:02

@author: YinongLong
@file: utility.py

处理推荐使用的数据集
"""
from __future__ import print_function

import os
import sys

if sys.platform == 'darwin':
    DATA_DIR = '/Users/Yinong/Downloads/ml-100k'
else:
    DATA_DIR = 'E:/fireFoxDownload/ml-100k'

RATING_DATA = 'u1.base'
TEST_DATA = 'u1.test'
MOVIE_INFO = 'u.item'
RATING_PATH = os.path.join(DATA_DIR, RATING_DATA)
TEST_PATH = os.path.join(DATA_DIR, TEST_DATA)
MOVIE_PATH = os.path.join(DATA_DIR, MOVIE_INFO)


def _load_data(func):
    """
    读取需要处理的数据
    :return: 
    """

    def load_data_(file_path=func()):
        data = list()
        with open(file_path, 'rb') as data_file:
            for line in data_file:
                user_id, item_id, score, _ = line.strip().split()
                user_id, item_id, score = int(user_id), int(item_id), float(score)
                data.append((user_id, item_id, score))
        return data
    return load_data_


@_load_data
def load_test_data(file_path=TEST_PATH):
    return file_path


@_load_data
def load_train_data(file_path=RATING_PATH):
    return file_path


def load_movie_info(file_path=MOVIE_PATH):
    result = dict()
    with open(file_path, 'rb') as data_file:
        for line in data_file:
            information = line.strip().split('|')
            item_id = int(information[0])
            item_name = information[1]
            result[item_id] = item_name
    return result


def main():
    # data_dir = 'E:/fireFoxDownload/ml-100k'
    # data = load_train_data()
    # load_test_data(os.path.join(data_dir, 'u1.test'))
    # print(len(data))
    load_movie_info()
    pass


if __name__ == '__main__':
    main()
