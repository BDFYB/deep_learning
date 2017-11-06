#encoding=utf-8
import tensorflow as tf
import os
import imdb_movie_reviews
import embedding
import numpy as np
#废弃。一共25k数据，5k tf_records文件就占10G
#

def preprocessd_data(iterator, length, embedding):
    #iter() 返回iterator对象，就是有个next()方法的对象
    #参数collection应是一个容器，支持迭代协议(即定义有__iter__()函数)，
    #或者支持序列访问协议（即定义有__getitem__()函数），否则会返回TypeError异常
    iterator = iter(iterator)
    while True:
        data = np.zeros((length, embedding.dimensions))
        target = np.zeros((2))
        text, label = next(iterator)
        data = embedding(text)
        target = [1, 0] if label else [0, 1]
        yield (data, target)


if __name__ == "__main__":
    IMDB_DOWNLOAD_DIR = './imdb'
    IMDB_DATA_DIR = './imdb/aclImdb'
    IMDB_TFR_DIR = './imdb/TFR'
    WIKI_VOCAB_DIR = '../1_wikipedia/wikipedia'
    WIKI_EMBED_DIR = '../1_wikipedia/wikipedia'

    if os.path.isdir(IMDB_TFR_DIR):
        os.system("rm -rf %s" % IMDB_TFR_DIR)
    os.makedirs(IMDB_TFR_DIR)

    #reviews = imdb_movie_reviews.ImdbMovieReviewsDownloadVersion(IMDB_DOWNLOAD_DIR)
    reviews = imdb_movie_reviews.ImdbMovieReviews(IMDB_DATA_DIR)
    length = max(len(x[0]) for x in reviews)

    embedding = embedding.Embedding(
        WIKI_VOCAB_DIR + '/vocabulary.bz2',
        WIKI_EMBED_DIR + '/embeddings.npy', length)

    batch = preprocessd_data(reviews, length, embedding)

    writer = None
    for index, single_record in enumerate(batch):
        #TFRecord
        if index % 2500 == 0:
            print("current steps:%s" % index)
            if writer:
                writer.close()
            tf_file_name = IMDB_TFR_DIR + "/review%s.tfr" % index
            writer = tf.python_io.TFRecordWriter(tf_file_name)

        data_list = np.array(single_record[0]).tobytes()
        data_label = np.array(single_record[1]).tobytes()

        #example协议
        example = tf.train.Example(features = tf.train.Features(feature={
                "label": tf.train.Feature(bytes_list = tf.train.BytesList(value=[data_label])),
                "review": tf.train.Feature(bytes_list = tf.train.BytesList(value=[data_list])),}))

        writer.write(example.SerializeToString())







