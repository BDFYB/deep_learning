import tensorflow as tf 
from helpers import lazy_property

class BidirectionSequenceLabelingModule(object):

    def __init__(self, data, target, params):
        self.data = data
        self.target = target
        self.params = params
        #这几个不写不行！会报error!
        self.length
        self.prediction
        self.loss
        self.error
        self.optimize

    @lazy_property
    def length(self):
        length = tf.reduce_sum(tf.reduce_max(self.data, reduction_indices = 2), reduction_indices = 1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        output = self._bidirectional_rnn(self.data, self.length)
        class_size = int(self.target.get_shape()[2])
        prediction = self._shared_softmax(output, class_size)
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
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2)) 
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)

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

    def _bidirectional_rnn(self, data, length):
        forward_output, _ = tf.nn.dynamic_rnn(
            cell = self.params.rnn_cell(self.params.rnn_hidden_layer_size),
            inputs = data, 
            sequence_length = length, 
            dtype = tf.float32,
            scope = 'rnn_forward'
        )
        backward_output, _ = tf.nn.dynamic_rnn(
            cell = self.params.rnn_cell(self.params.rnn_hidden_layer_size),
            inputs = tf.reverse_sequence(data, length, seq_dim = 1),
            sequence_length = length,
            dtype = tf.float32,
            scope = 'rnn_backward',
        )
        backward_output = tf.reverse_sequence(backward_output, length, seq_dim = 1)
        output = tf.concat([forward_output, backward_output], 2)
        return output

    def _shared_softmax(self, data, class_size):
        double_rnn_hidden_layer_size = int(data.get_shape()[2])
        sequence_length = int(data.get_shape()[1])

        flattern_data = tf.reshape(data, [-1, double_rnn_hidden_layer_size])

        weight = tf.Variable(tf.truncated_normal([double_rnn_hidden_layer_size, class_size], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[class_size]))

        fully_connected = tf.nn.softmax(tf.matmul(flattern_data, weight) + bias)

        result = tf.reshape(fully_connected, [-1, sequence_length, class_size], name="prediction")
        return result







