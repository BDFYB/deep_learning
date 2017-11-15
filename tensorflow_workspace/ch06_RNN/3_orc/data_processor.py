import csv
import os
import gzip
import numpy as np
from helpers import download

"""
ocr Optical Character Recognition
类似图像中文描述，需要cnn + rnn
数据已准备好，无需cnn，类似featuremap了

ocr data 每行文件前几列意义
1 id: each letter is assigned a unique integer id
2 letter: a-z
3 next_id: id for next letter in the word, -1 if last letter
4 word_id: each word is assigned a unique integer id (not used)
5 position: position of letter in the word (not used)
6 fold: 0-9 -- cross-validation fold
7 p_i_j: 0/1 -- value of pixel in row i, column j, 正式数据开始

"""
class OcrData(object):

    def __init__(self, use_local_file = False):
        self.data_url = 'http://ai.stanford.edu/~btaskar/ocr/letter.data.gz'
        self.default_data_file = "./ocr_data/letter.data"
        self.default_data_dir = "./ocr_data"
        self.use_local_file = use_local_file
        if use_local_file:
            filename = self.default_data_file
            if not os.path.isfile(self.default_data_file):
                raise Exception("File not found!")
        else:
            filename = download(self.data_url, self.default_data_dir)
        lines = self._readlines(filename)
        #将字母按照预处理的数据格式拼接成单词
        data, target = self._parse(lines)
        #data size: 6877 * n * 16 * 8
        #target size: 6877 * n
        #n : 一个单词中字母个数
        """
        print(np.array(data).shape)
        #这种第二维度个数不一样的，无法一次性打印出
        print(np.array(data[2222]).shape)
        print(np.array(target).shape)
        print(np.array(target[5555]).shape)
        print([len(x) for x in target])
        """

        #将数据及target填充成等长（最大长度）
        self.data, self.target = self._pad(data, target)
        #padding 后 size: data (6877, 14, 16, 8) target: (6877, 14)
        #print(np.array(self.data).shape)
        #print(np.array(self.target).shape)


    def _readlines(self, filename):
        if self.use_local_file:
            with open(filename) as file_:
                reader = csv.reader(file_, delimiter='\t')
                lines = list(reader)
                return lines
        else:
            with gzip.open(filename, 'rt') as file_:
                reader = csv.reader(file_, delimiter='\t')
                lines = list(reader)
                return lines


    def _parse(self, lines):
        #首先按照每行第一个数字（标识顺序的）进行排序 next_id一般为下一行字母，-1为单词结束
        #sorted key指定一个接收一个参数的函数，这个函数用于从每个元素中提取一个用于比较的关键字。
        lines = sorted(lines, key=lambda x: int(x[0]))

        data, target = [], []
        next_ = None
        for line in lines:
            if not next_:
                data.append([])
                target.append([])
            else:
                assert next_ == int(line[0])
            next_ = int(line[2]) if int(line[2]) > -1 else None
            pixels = np.array([int(x) for x in line[6:134]])
            pixels = pixels.reshape((16, 8))
            data[-1].append(pixels)
            target[-1].append(line[1])
            
        return data, target


    def _pad(self, data, target):
        max_length = max(len(x) for x in target)
        padding = np.zeros((16, 8))
        data = [x + ([padding] * (max_length - len(x))) for x in data]
        target = [x + ([''] * (max_length - len(x))) for x in target]
        return np.array(data), np.array(target)


if __name__ == "__main__":
    data_set = OcrData(use_local_file = True)
