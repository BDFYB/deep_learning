#encoding=utf-8

import bz2
import numpy as np

class Embedding(object):

    def __init__(self, vocabulary_path, embedding_record_path, length):
        #加载embedding
        self._embedding = np.load(embedding_record_path)
        #加载字典
        with bz2.open(vocabulary_path, 'rt') as file_:
            self._vocabulary = {k.strip(): i for i, k in enumerate(file_)}
        self._length = length

    #实现__call__函数，这个类型就成为可调用的
    def __call__(self, sequence):
        data = np.zeros((self._length, self._embedding.shape[1]))
        #返回单词在字典里的索引位置，作为embedding_lookup参数
        #dict.get(key, default=None)则如果不存在则返回一个默认值
        indices = [self._vocabulary.get(x, 0) for x in sequence]
        #根据索引值获取embedding
        embedded = self._embedding[indices]
        data[:len(sequence)] = embedded
        return data

    @property
    def dimensions(self):
        return self._embedding.shape[1]
