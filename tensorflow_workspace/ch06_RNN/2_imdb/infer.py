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

    with tf.gfile.GFile(FORZEN_GRAPH, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read()) 
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(  
            graph_def,   
            input_map = None,   
            return_elements = None,   
            name = "frozen_module",   
            op_dict = None,   
            producer_op_list = None  
        )
       
    """
    #查看计算图所有op 
    for op in graph.get_operations():  
        print(op.name,op.values()) 
    """

    #数据准备
    train_data = imdb_movie_reviews.ImdbMovieReviews(IMDB_DATA_DIR)
    length = max(len(x[0]) for x in train_data)

    test_data = imdb_movie_reviews.ImdbMovieReviews(IMDB_DATA_DIR, infer = True)
    embedding = embedding.Embedding(
        WIKI_VOCAB_DIR + '/vocabulary.bz2',
        WIKI_EMBED_DIR + '/embeddings.npy', length)
    batchs = preprocessd_batch(test_data, length, embedding, batch_size=1)

    with tf.Session(graph=graph) as sess:
        input_data_holder = graph.get_tensor_by_name("frozen_module/input_data:0")
        input_label_holder = graph.get_tensor_by_name("frozen_module/input_target:0")
        softmax = graph.get_tensor_by_name("frozen_module/softmax_prediction:0")

        for index, batch in enumerate(batchs):
            feed_data = {
                input_data_holder: batch[0],
                input_label_holder: batch[1],
            }
            softmax_result = sess.run([softmax], feed_data)
            print(batch[1])
            print(softmax_result)
            loss = -tf.reduce_sum(batch[1] * tf.log(softmax_result[0]))
            print(sess.run(loss))

