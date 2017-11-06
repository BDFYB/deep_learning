import random


def Skipgrams(pages, max_context):
    """Form training pairs according to the skip-gram model."""
    # max_context:上下文最长数
    for words in pages:
        """words:一个用index表示的list, eg:[2602, 5302, 0, 3,......]
        """
        """
        对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），
        enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        enumerate多用于在for循环中得到计数
        """
        for index, current in enumerate(words):
            #print("Skipgrams, index: %s, current: %s" % (index, current))
            context = random.randint(1, max_context)
            for target in words[max(0, index - context): index]:
                yield current, target
            for target in words[index + 1: index + context + 1]:
                yield current, target
