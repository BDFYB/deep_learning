#!/usr/bin/python 
# -*- coding: utf-8 -*-

import numpy as np
import os
import tensorflow as tf

W = tf.Variable(tf.zeros([4, 3]), name="weights")
b = tf.Variable(tf.zeros([3], name="bias"))

def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)

def inputs():
    batch_size = 50
    record_defaults = [[0.0], [0.0], [0.0], [0.0], [""]]
    """
    filename_queue = tf.train.string_input_producer(['iris.data'])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    sepal_length, sepal_width, petal_length, petal_width, label = tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)
    """

    sepal_length, sepal_width, petal_length, petal_width, label = \
            read_csv(batch_size, "iris.data", record_defaults)
    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([tf.equal(label, "Iris-setosa"), tf.equal(label, ["Iris-versicolor"]), tf.equal(label, ["Iris-virginica"])])), 0))
    features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))
    return features, label_number

def train(total_loss):
    #use 
    learning_rate = 0.01    
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def combine_inputs(X):
    return tf.matmul(X, W) + b

def inference(X):
    return tf.nn.softmax(combine_inputs(X))
    
def loss(X, Y):
    #use cross entropy
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=combine_inputs(X)))

def evaluate(sess, X, Y):
    predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)
    #准确率
    print sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))

if __name__ == "__main__":
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        #tf.initialize_all_variables().run()

        X, Y = inputs()

        total_loss = loss(X, Y)
        train_op = train(total_loss)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # actual training loop
        training_steps = 1000
        for step in range(training_steps):
            sess.run([train_op])
            if step % 300 == 0:
                print "loss: ", sess.run([total_loss])

        evaluate(sess, X, Y)
        writer = tf.summary.FileWriter('./tensorboard_softmax_regression', sess.graph)
        #writer = tf.summary.FileWriter('./tensorboard_logistic_regression', sess.graph)
        writer.close()

        import time
        time.sleep(5)
        coord.request_stop()
        coord.join(threads)
        sess.close()
