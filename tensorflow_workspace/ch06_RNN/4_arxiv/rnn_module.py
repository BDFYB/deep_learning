import tensorflow as tf 
from helpers import lazy_property

class PredictionModule(object):

    def __init__(self, params, sequence, initial=None):
        self.params = params
        self.sequence_batch = sequence
        self.initial = initial
        self.length
        self.prediction
        self.state
        self.loss
        self.error
        self.logprob
        self.optimize

    @lazy_property
    def data(self):
        #输入序列的0-(len-1)作为输入
        charactor_length = int(self.sequence_batch.get_shape()[1])
        return tf.slice(self.sequence_batch, (0, 0, 0), (-1, charactor_length - 1, -1))

    @lazy_property
    def target(self):
        #输入序列的1-(len)作为标签
        return tf.slice(self.sequence_batch, (0, 1, 0), (-1, -1, -1))

    @lazy_property
    def mask(self):
        #主要是为了去除padding为0的项
        return tf.reduce_max(tf.abs(self.data), reduction_indices=2)

    @lazy_property
    def length(self):
        return tf.reduce_sum(self.mask, reduction_indices = 1)

    @lazy_property
    def prediction(self):
        prediction, _ = self.network
        return prediction

    @lazy_property
    def state(self):
        _, state = self.network
        return state

    @lazy_property
    def network(self):
        cell_layer_1 = self.params.rnn_cell(self.params.rnn_hidden_layer_size)
        cell_layer_2 = self.params.rnn_cell(self.params.rnn_hidden_layer_size)
        #不可以乘以2，这样两个cell是一个实例，共享内部变量会有数据维度问题
        #cell_combine = tf.nn.rnn_cell.MultiRNNCell([cell_layer_1] * self.params.rnn_layers)
        cell_combine = tf.nn.rnn_cell.MultiRNNCell([cell_layer_1, cell_layer_2])
        hidden, state = tf.nn.dynamic_rnn(
            inputs=self.data,
            cell=cell_combine,
            dtype=tf.float32,
            initial_state=self.initial,
            sequence_length=self.length)
        #batch_size * charactor_length * rnn_hidden_layer_size
        prediction = self._shared_softmax(hidden, int(self.target.get_shape()[2]))
        return prediction, state

    @lazy_property
    def loss(self):
        #softmax
        prediction = tf.clip_by_value(self.prediction, 1e-10, 1.0)
        cross_entropy = self.target * tf.log(prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices = 2)
        cross_entropy = cross_entropy * self.mask
        length = tf.reduce_sum(self.length, 0)
        #对一个数据的所有帧取平均
        avr = tf.reduce_sum(cross_entropy, reduction_indices = 1) / length
        #最后一步对批数据中的样本取平均
        cross_entropy = tf.reduce_mean(avr, name="train_loss")
        return cross_entropy

    @lazy_property
    def optimize(self):
        #outtime!optimizer是可以直接传梯度剪裁参数的
        gradient = self.params.optimizer.compute_gradients(self.loss)
        if self.params.gradient_clipping:
            limit = self.params.gradient_clipping
            gradient = [
                (tf.clip_by_value(g, -limit, limit), v)
                if g is not None else (None, v)
                for g, v in gradient]
        optimize = self.params.optimizer.apply_gradients(gradient)
        return optimize

    @lazy_property
    def error(self):
        error = tf.not_equal(tf.argmax(self.prediction, 2), tf.argmax(self.target, 2))
        error = tf.cast(error, tf.float32)
        #error
        error = error * self.mask
        length = tf.reduce_sum(self.length, 0)
        #对一个数据的所有帧取平均
        avr = tf.reduce_sum(error, reduction_indices = 1) / length
        #最后一步对批数据中的样本取平均
        error = tf.reduce_mean(avr, name="train_error")
        return error


    @lazy_property
    def logprob(self):
        logprob = tf.multiply(self.prediction, self.target)
        logprob = tf.reduce_max(logprob, reduction_indices=2)
        logprob = tf.log(tf.clip_by_value(logprob, 1e-10, 1.0)) / tf.log(2.0)
        return self._average(logprob)


    def _shared_softmax(self, data, output_size):
        #这里需要用int转换一下！！！
        label_length = int(data.get_shape()[1])
        input_size = int(data.get_shape()[2])
        weight = tf.Variable(tf.truncated_normal([input_size, output_size], stddev = 0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[output_size]))
        rnn_reshaped_output = tf.reshape(data, [-1, input_size])
        output = tf.matmul(rnn_reshaped_output, weight) + bias
        prediction = tf.nn.softmax(output)
        return tf.reshape(prediction, [-1, label_length, output_size])

    def _average(self, data):
        data = data * self.mask
        length = tf.reduce_sum(self.length, 0)
        #对一个数据的所有帧取平均
        avr = tf.reduce_sum(data, reduction_indices = 1) / length
        #最后一步对批数据中的样本取平均
        return tf.reduce_mean(avr)
