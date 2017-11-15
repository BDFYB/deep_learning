import tensorflow as tf 
from helpers import lazy_property

class SequenceLabelingModule(object):

    def __init__(self, data, target, params):
        self.data = data
        self.target = target
        self.params = params
        #这几个不写不行！会报error!
        self.prediction
        self.loss
        self.error
        self.optimize

    @lazy_property
    def length(self):
        #获取每个句子的有效长度
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        # rnn_layer
        output, _ = tf.nn.dynamic_rnn(
            self.params.rnn_cell(self.params.rnn_hidden_layer_size), 
            self.data, 
            sequence_length=self.length, 
            dtype=tf.float32
        )

        # softmax_layer
        max_length = int(self.target.get_shape()[1])
        class_size = int(self.target.get_shape()[2])
        weight = tf.Variable(tf.truncated_normal(
            [self.params.rnn_hidden_layer_size, class_size], stddev = 0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[class_size]))
        #先将batch_size*sequence_lenth扁平化，共享softmax层
        output = tf.reshape(output, [-1, self.params.rnn_hidden_layer_size])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        #反扁平化，恢复序列结构
        prediction = tf.reshape(prediction, [-1, max_length, class_size], name = "prediction")
        return prediction
        
    @lazy_property
    def loss(self):
        #log：对应项取log
        #相乘：对应项相乘
        cross_entropy = self.target * tf.log(self.prediction)
        #对reduction_indices维合并，其他维保证结构不变
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices = 2))
        cross_entropy = cross_entropy * mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices = 1)
        cross_entropy = cross_entropy / tf.cast(self.length, tf.float32)
        loss = tf.reduce_mean(cross_entropy, name = "train_loss")
        return loss

    @lazy_property
    def optimize(self):
        gradient = self.params.optimizer.compute_gradients(self.loss)
        try:
            limit = self.params.gradient_clipping
            gradient = [
                (tf.clip_by_value(g, -limit, limit), v)
                if g is not None else (None, v)
                for g, v in gradient]
        except AttributeError:
            print('No gradient clipping parameter specified.')
        optimize = self.params.optimizer.apply_gradients(gradient)
        return optimize

    @lazy_property
    def error(self):
        #argmax: 按维度比较大小，返回最大元素下标
        mistakes = tf.not_equal(tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)










