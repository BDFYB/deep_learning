#encoding=utf-8
import os
import glob
import tarfile
import re
import random
from helpers import download

"""
影评数据集已下载解压完成版本
"""
class ImdbMovieReviews(object):
    DEFAULT_DIR = "/Users/baidu/AI/deep_learning/tensorflow_workspace/ch06_RNN/2_imdb/imdb/aclImdb"
    TOKEN_REGEX = re.compile(r'[A-Za-z]+|[!?.:,()]')

    def __init__(self, review_data_top_dir=None, infer=False):
        top_dir = review_data_top_dir or type(self).DEFAULT_DIR
        if infer:
            self._pos_data_dir = top_dir + '/test/pos/'
            self._neg_data_dir = top_dir + '/test/neg/'     
        else:       
            self._pos_data_dir = top_dir + '/train/pos/'
            self._neg_data_dir = top_dir + '/train/neg/'

        if not os.path.isdir(self._pos_data_dir) or not os.path.isdir(self._neg_data_dir):
            raise Exception("data dir not exsists! please use this function with download version")

    def __iter__(self):
        #这么训练实践上有问题！全会输出[0, 1]（都以最后一批训练的为准了，典型的过拟合）
        """
        for filename in os.listdir(self._pos_data_dir):
            yield self._read(self._pos_data_dir+filename), True
        for filename in os.listdir(self._neg_data_dir):
            yield self._read(self._neg_data_dir+filename), False
        """
        pos_file_list = os.listdir(self._pos_data_dir)
        neg_file_list = os.listdir(self._neg_data_dir)
        total_file_size = len(pos_file_list) + len(neg_file_list)
        for index in range(total_file_size):
            if random.randint(0, 1) == 0:
                if (len(neg_file_list) == 0):
                    print("neg_file_list DONE!")
                    continue
                file_name = self._neg_data_dir+neg_file_list.pop()
                #print(file_name)
                yield self._read(file_name), False
            else:
                if (len(pos_file_list) == 0):
                    print("pos_file_list DONE!")
                    continue
                file_name = self._pos_data_dir+pos_file_list.pop()
                #print(file_name)
                yield self._read(file_name), True


    def _read(self, filename):
        with open(filename, 'r') as file:
            data = file.read()
            data = type(self).TOKEN_REGEX.findall(data)
            data = [x.lower() for x in data]
            return data

"""
需要下载原始影评数据集的版本
"""
class ImdbMovieReviewsDownloadVersion(object):
    DEFAULT_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    TOKEN_REGEX = re.compile(r'[A-Za-z]+|[!?.:,()]')

    def __init__(self, cache_dir, url=None):
        self._cache_dir = cache_dir
        #type(self)返回类型名
        self._url = url or type(self).DEFAULT_URL

    def __iter__(self):
        filepath = download(self._url, self._cache_dir)
        with tarfile.open(filepath) as archive:
            for filename in archive.getnames():
                if filename.startswith('aclImdb/train/pos/'):
                    yield self._read(archive, filename), True
                elif filename.startswith('aclImdb/train/neg/'):
                    yield self._read(archive, filename), False

    def _read(self, archive, filename):
        with archive.extractfile(filename) as file_:
            data = file_.read().decode('utf-8')
            data = type(self).TOKEN_REGEX.findall(data)
            data = [x.lower() for x in data]
            return data
