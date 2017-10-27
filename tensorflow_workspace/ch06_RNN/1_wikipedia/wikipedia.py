# -*- coding: utf-8 -*-

import bz2
import collections
import os
import re

class Wikipedia():

    def __init__(self, url, cache_dir, vocabulary_size = 10000):
        # 将～等用用户的家目录进行替换，如~/tmp替换为/home/work/tmp 
        self._cache_dir = os.path.expanduser(cache_dir)
        self._pages_path = os.path.join(self._cache_dir, 'pages.bz2')
        self._vocabulary_path = os.path.join(self._cache_dir, 'vocabulary.bz2')
        if not os.path.isfile(self._pages_path):
            print('Read pages')
            self._read_pages(url)
        if not os.path.isfile(self._vocabulary_path):
            print('Build vocabulary')
            self._build_vocabulary(vocabulary_size)
        with bz2.open(self._vocabulary_path, 'rt') as vocabulary:
            print('Read vocabulary')
            self._vocabulary = [x.strip() for x in vocabulary]
        self._indices = {x: i for i, x in enumerate(self._vocabulary)}

    def __iter__(self):
        """Iterate over pages represented as lists of word indices."""
        #实现了__iter__方法的类是可迭代的，__iter__返回可迭代对象。当使用for 循环迭代类实例时，将会调用
        #本方法返回的可迭代对象中的next方法，在这里面就是words的next()方法，就是一个list。
        #这么实现的好处是无需将整个words存储到内存中，每次迭代时再产生数据
        #实现了类似tf的队列
        with bz2.open(self._pages_path, 'rt') as pages:
            for page in pages:
                words = page.strip().split()
                words = [self.encode(x) for x in words]
                yield words

    @property
    def vocabulary_size(self):
        return len(self._vocabulary)

    def encode(self, word):
        """Get the vocabulary index of a string word."""
        return self._indices.get(word, 0)

    def decode(self, index):
        """Get back the string word from a vocabulary index."""
        return self._vocabulary[index]

    def _read_pages(self, url):
        pass

    def _build_vocabulary(self, vocabulary_size):
        pass

    @classmethod
    def _tokenize(cls, page):
        pass