#encoding=utf-8
import tensorflow as tf
import imdb_movie_reviews
import embedding
import rnn_module
from helpers import attribute_dictionary
import numpy as np

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
        yield data, target


if __name__ == "__main__":
    IMDB_DOWNLOAD_DIR = './imdb'
    WIKI_VOCAB_DIR = '../1_wikipedia/wikipedia'
    WIKI_EMBED_DIR = '../1_wikipedia/wikipedia'
    params = attribute_dictionary.AttrDict(
            rnn_cell = tf.nn.rnn_cell.GRUCell,
            rnn_hidden = 300,
            optimizer = tf.train.RMSPropOptimizer(0.002),
            batch_size = 20,
            gradient_clipping = 0,
        )
    reviews = imdb_movie_reviews.ImdbMovieReviews(IMDB_DOWNLOAD_DIR)
    length = max(len(x[0]) for x in reviews)

    embedding = embedding.Embedding(
        WIKI_VOCAB_DIR + '/vocabulary.bz2',
        WIKI_EMBED_DIR + '/embeddings.npy', length)

    batchs = preprocessd_batch(reviews, length, embedding, params.batch_size)
    data = tf.placeholder(tf.float32, [None, length, embedding.dimensions])
    target = tf.placeholder(tf.float32, [None, 2])
    model = rnn_module.LearningModel(data, target, params)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, batch in enumerate(batchs):
            feed_data = {
                data: batch[0],
                target: batch[1],
            }
            error, _ = sess.run([model.error, model.optimizer], feed_data)
            if index % 1000 == 0:
                print('{}: {:3.1f}%'.format(index + 1, 100 * error))