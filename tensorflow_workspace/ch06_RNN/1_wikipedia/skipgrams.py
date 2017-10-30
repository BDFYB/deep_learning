import random


def Skipgrams(pages, max_context):
    """Form training pairs according to the skip-gram model."""
    for words in pages:
        """
        对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），
        enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        enumerate多用于在for循环中得到计数
        """
        for index, current in enumerate(words):
            context = random.randint(1, max_context)
            for target in words[max(0, index - context): index]:
                yield current, target
            for target in words[index + 1: index + context + 1]:
                yield current, target
