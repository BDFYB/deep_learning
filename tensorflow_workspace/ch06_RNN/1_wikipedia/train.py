from helpers import attribute_dictionary
import embeddingModule
import wikipedia
import skipgrams
import batched
import collections
import tensorflow as tf
import numpy as np

WIKI_DOWNLOAD_DIR = './wikipedia'

if __name__ == "__main__":
    params = attribute_dictionary.AttrDict(
        vocabulary_size = 10000,
        max_context = 10,
        embedding_size = 200,
        contrasitive_examples = 100,
        learning_rate = 0.5,
        momentum = 0.5,
        batch_size = 1000,)
    data = tf.placeholder(tf.int32, [None])
    target = tf.placeholder(tf.int32, [None])
    model = embeddingModule.EmbeddingModule(data, target, params)

    corpus = wikipedia.Wikipedia(
        'https://dumps.wikimedia.org/enwiki/20170701/'
        'enwiki-20170701-pages-meta-current1.xml-p10p30303.bz2',
        WIKI_DOWNLOAD_DIR,
        params.vocabulary_size)
    examples = skipgrams.Skipgrams(corpus, params.max_context)
    batches = batched.Batched(examples, params.batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        average = collections.deque(maxlen=100)
        for index, batch in enumerate(batches):
            feed_dict = {
                data: batch[0],
                target: batch[1],
            }
            cost, _ = sess.run([model.cost, model.optimize], feed_dict)
            average.append(cost)
            if index%10000 == 0:
                print('{}: {:5.1f}'.format(index + 1, sum(average) / len(average)))
                if index > 100000:
                    break

        embeddings = sess.run(model.embeddings)
        np.save(WIKI_DOWNLOAD_DIR + '/embeddings.npy', embeddings)
        