#encoding=utf-8
import tensorflow as tf
import imdb_movie_reviews
import embedding
import rnn_module
import os
from helpers import attribute_dictionary
from tensorflow.python.framework import graph_util
import numpy as np

#直接处理生数据，在训练前同时处理数据版本

def preprocessd_batch(iterator, length, embedding, batch_size):
    #iter() 返回iterator对象，就是有个next()方法的对象
    #参数collection应是一个容器，支持迭代协议(即定义有__iter__()函数)，
    #或者支持序列访问协议（即定义有__getitem__()函数），否则会返回TypeError异常
    iterator = iter(iterator)
    while True:
        data = np.zeros((batch_size, length, embedding.dimensions))
        target = np.zeros((batch_size, 2))
        for index in range(batch_size):
            text, label = next(iterator)
            data[index] = embedding(text)
            target[index] = [1, 0] if label else [0, 1]
        yield (data, target)


if __name__ == "__main__":
    IMDB_DOWNLOAD_DIR = './imdb'
    IMDB_DATA_DIR = './imdb/aclImdb'
    WIKI_VOCAB_DIR = '../1_wikipedia/wikipedia'
    WIKI_EMBED_DIR = '../1_wikipedia/wikipedia'
    SUMMARY_DIR = './summary'
    FORZEN_GRAPH = './frozen_graph/freeze_module.pb'
    params = attribute_dictionary.AttrDict(
            rnn_cell = tf.nn.rnn_cell.GRUCell,
            rnn_hidden = 300,
            optimizer = tf.train.RMSPropOptimizer(0.002),
            batch_size = 20,
            gradient_clipping = 0,
        )

    if os.path.isdir(SUMMARY_DIR):
        os.system("rm -rf %s" % SUMMARY_DIR)

    if os.path.isfile(FORZEN_GRAPH):
        os.system("rm %s" % FORZEN_GRAPH)

    #reviews = imdb_movie_reviews.ImdbMovieReviewsDownloadVersion(IMDB_DOWNLOAD_DIR)
    reviews = imdb_movie_reviews.ImdbMovieReviews(IMDB_DATA_DIR)
    length = max(len(x[0]) for x in reviews)

    embedding = embedding.Embedding(
        WIKI_VOCAB_DIR + '/vocabulary.bz2',
        WIKI_EMBED_DIR + '/embeddings.npy', length)

    batchs = preprocessd_batch(reviews, length, embedding, params.batch_size)
    data = tf.placeholder(tf.float32, [None, length, embedding.dimensions], name="input_data")
    target = tf.placeholder(tf.float32, [None, 2], name="input_target")
    model = rnn_module.LearningModel(data, target, params)

    err_summary = tf.summary.scalar("train_error", model.error)
    summary_op = tf.summary.merge([err_summary])
    loss_summary = tf.summary.scalar("train_loss", model.cost)
    summary_op2 = tf.summary.merge([loss_summary])
    total_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

        for index, batch in enumerate(batchs):
            feed_data = {
                data: batch[0],
                target: batch[1],
            }
            error, _, g_summary = sess.run([model.error, model.optimizer, total_summary_op], feed_data)
            train_summary_writer.add_summary(g_summary, index)
            if index % 50 == 0:
                print('{}: {:3.1f}%'.format(index + 1, 100 * error))
                train_summary_writer.flush()

        #freeze result and graph
        cur_graph = tf.get_default_graph()
        input_graph_def = cur_graph.as_graph_def()
        output_node_names = ["softmax_prediction", "input_target", "loss"]
        output_graph_def = graph_util.convert_variables_to_constants(  
            sess,   
            input_graph_def,   
            output_node_names, # We split on comma for convenience  
        ) 
        # Finally we serialize and dump the output graph to the filesystem  
        with tf.gfile.GFile(FORZEN_GRAPH, "wb") as f:  
            f.write(output_graph_def.SerializeToString())  
            print("%d ops in the final graph." % len(output_graph_def.node)) 
