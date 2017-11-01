#encoding=utf-8
import tensorflow as tf 
from helpers import lazy_property

class LearningModel(object):

    def __init__(self, data, target, params):
        self.data = data
        self.target = target
        self.params = params
        self.prediction    
        self.cost
        self.optimizer
        self.error

    #装饰器是一个函数，其主要用途是包装另一个函数或类。这种包装的首要目的是透明地修改或增强被包装对象的行为。
    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        #rnn
        output, _ = tf.nn.dynamic_rnn(
            self.params.rnn_cell(self.params.rnn_hidden),
            self.data,
            dtype = tf.float32,
            sequence_length = self.length)
        last = self._last_relevant(output, self.length)
        #softmax
        num_classes = int(self.target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal(
            [self.params.rnn_hidden, num_classes], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimizer(self):
        gradient = self.params.optimizer.compute_gradients(self.cost)
        if self.params.gradient_clipping == 1:
            limit = self.params.gradient_clipping
            gradient = [
                (tf.clip_by_value(g, -limit, limit), v)
                if g is not None else (None, v)
                for g, v in gradient
            ]
        optimizer = self.params.optimizer.apply_gradients(gradient)
        return optimizer

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length -1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant