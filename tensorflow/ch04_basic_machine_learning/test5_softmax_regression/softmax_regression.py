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
    sepal_length, sepal_width, petal_length, petal_width, label = \
            read_csv(1, "iris.data", [[0.0], [0.0], [0.0], [0.0], [""]])
    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.pack([tf.equal(label, "Iris-setosa"), tf.equal(label, ["Iris-versicolor"]), tf.equal(label, ["Iris-virginica"])])), 0))
    features = tf.transpose(tf.pack([sepal_length, sepal_width, petal_length, petal_width]))
    print features

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
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(combine_inputs(X), Y))

def evaluate(sess, X, Y):
    predicted = tf.cast(tf.arg_max(inference(X), 1))
    #准确率
    print sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))

if __name__ == "__main__":
    with tf.Session() as sess:

        tf.initialize_all_variables().run()

        X, Y = inputs()

        total_loss = loss(X, Y)
        train_op = train(total_loss)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # actual training loop
        training_steps = 100
        for step in range(training_steps):
            sess.run([train_op])
            if step % 300 == 0:
                print "loss: ", sess.run([total_loss])

        evaluate(sess, X, Y)
        writer = tf.train.SummaryWriter('./tensorboard_logistic_regression', sess.graph)
        writer.close()

        import time
        time.sleep(5)
        coord.request_stop()
        coord.join(threads)
        sess.close()
