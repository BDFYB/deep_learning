import tensorflow as tf
from helpers import lazy_property

class EmbeddingModule(object):

    def __init__(self, data, target, params):
        self.data = data
        self.target = target
        self.params = params
        self.embeddings
        self.cost
        self.optimize

    """
    Python 对象的延迟初始化是指，当它第一次被创建时才进行初始化，
    或者保存第一次创建的结果，然后每次调用的时候直接返回该结果。
    """
    @lazy_property
    def embeddings(self):
        initial = tf.random_uniform([
                self.params.vocabulary_size,
                self.params.embedding_size,
            ], -1.0, 1.0)
        return tf.Variable(initial)

    @lazy_property
    def optimize(self):
        optimizer = tf.train.MomentumOptimizer(
            self.params.learning_rate, self.params.momentum)
        return optimizer.minimize(self.cost)

    @lazy_property
    def cost(self):
        embedded = tf.nn.embedding_lookup(self.embeddings, self.data)
        weight = tf.Variable(tf.truncated_normal(
                                [self.params.vocabulary_size, self.params.embedding_size],
                                stddev = 1.0/self.params.embedding_size ** 0.5))
        bias = tf.Variable(tf.zeros([self.params.vocabulary_size]))
        target = tf.expand_dims(self.target, 1)
        return tf.reduce_mean(tf.nn.nce_loss(
            weight, bias, target, embedded, self.params.contrasitive_examples, self.params.vocabulary_size))
