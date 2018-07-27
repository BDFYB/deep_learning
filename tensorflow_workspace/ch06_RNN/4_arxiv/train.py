import arxiv_abstract
import data_processor
import rnn_module
import tensorflow as tf
import time
import os

from helpers import AttrDict


def get_params():
    checkpoint_dir = './summary'
    #每条数据的单词个数
    max_length = 50
    sampling_temperature = 0.7
    rnn_cell = tf.nn.rnn_cell.GRUCell
    rnn_hidden_layer_size = 200
    #size 适配MultiRNN
    #rnn_hidden_layer_size = 83
    rnn_layers = 2
    learning_rate = 0.002
    optimizer = tf.train.AdamOptimizer(0.002)
    gradient_clipping = 5
    batch_size = 100
    epochs = 20
    epoch_size = 200
    #print(locals())
    #locals返回当前作用域 的所有局部变量的变量名:变量值组成的字典。
    # **locals()在函数调用里的意思是将locals()返回的字典解包成key:value传递给对应函数
    return AttrDict(**locals())


if __name__ == "__main__":
   
    SUMMARY_DIR = "./summary"
    FORZEN_GRAPH = "./frozen_graph/forzen_module.pb"

    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)

    if os.path.isfile(FORZEN_GRAPH):
        os.system("rm %s" % FORZEN_GRAPH)

    params = get_params()
    data_obj = arxiv_abstract.ArxivAbstracts(    
        cache_dir = './arxiv',
        categories = [
            'Machine Learning',
            'Neural and Evolutionary Computing',
            'Optimization'
        ],
        keywords = [
            'neural',
            'network',
            'deep'
        ])

    texts = data_obj.test_data
    batched_data = data_processor.DataProcessor(texts, params.max_length, params.batch_size)
    #传1个参数：参数应是一个容器，支持迭代协议(即定义有__iter__()函数)，或者支持序列访问协议（即定义有__getitem__()函数），否则会返回TypeError异常。
    #返回：一个iterator对象，就是有个next()方法的对象。
    batch_data_iter = iter(batched_data)
    #一个batch的数据：(batch_size * params.max_length * onehot_size)
    onehot_size = len(data_processor.DataProcessor.VOCABULARY)
    input_data = tf.placeholder(tf.float32, [None, params.max_length, onehot_size], name="input")

    model = rnn_module.PredictionModule(params, input_data)

    err_summary = tf.summary.scalar("train_error", model.error)
    summary_op = tf.summary.merge([err_summary])
    loss_summary = tf.summary.scalar("train_loss", model.loss)
    summary_op2 = tf.summary.merge([loss_summary])
    total_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

        for index, batch_data in enumerate(batch_data_iter):
            #size:(batch_size, max_length, onehot_size)

            feed_dict = {
                input_data: batch_data
            }

            prediction, loss, error, log_prob, summary = sess.run([model.prediction, model.loss, model.error, model.logprob, total_summary_op], feed_dict)
            train_summary_writer.add_summary(summary, index)

            if (index % 50 == 0):
                print('{} step finished, loss: {:3.1f}%'.format(index + 1, 100 * loss))
                train_summary_writer.flush()






