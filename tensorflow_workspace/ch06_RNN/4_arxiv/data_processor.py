import random
import numpy as np

class DataProcessor(object):

    VOCABULARY = \
        " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
        "\\^_abcdefghijklmnopqrstuvwxyz{|}"

    def __init__(self, texts, length, batch_size):
        self.texts = texts
        self.length = length
        self.batch_size = batch_size
        self.lookup = {x: i for i, x in enumerate(self.VOCABULARY)}

    def __call__(self, texts):
        batch = np.zeros((len(texts), self.length, len(self.VOCABULARY)))
        for index, text in enumerate(texts):
            text = [x for x in text if x in self.lookup]
            assert 2 <= len(text) <= self.length
            for offset, character in enumerate(text):
                code = self.lookup[character]
                batch[index, offset, code] = 1
        #batch size: (batch_size * params.max_length * onehot_size)
        return batch

    def __iter__(self):
        windows = []
        #将所有句子按照params.max_length个字符做截断，每个截断是Windows的一个元素
        for text in self.texts:
            for i in range(0, len(text) - self.length + 1, self.length // 2):
                windows.append(text[i: i + self.length])

        # assert语句用来声明某个条件是真的。如果你非常确信某个条件，而你想要检验这一点，
        # 并且在它非真的时候引发一个错误，那么assert语句是应用在这种情形下的理想语句。
        # all(x)如果参数x对象的所有元素不为0、''、False或者x为空对象，则返回True，否则返回False
        assert all(len(x) == len(windows[0]) for x in windows)
        while True:
            random.shuffle(windows)
            #0到len()，间隔batch_size
            for i in range(0, len(windows), self.batch_size):
                batch = windows[i: i + self.batch_size]
                #调用__call__()方法，转成字符的独热编码
                yield self(batch)
