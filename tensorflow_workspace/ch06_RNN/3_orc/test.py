#encoding=utf-8
import tensorflow as tf
import data_processor 
import numpy as np
import batched
import time
from helpers import AttrDict
import rnn_module
import bidirection_rnn_module
from tensorflow.python.framework import graph_util
import os

def get_dataset():
    dataset = data_processor.OcrData(use_local_file = True)
    # shape of dataset.data:(6877, 14, 16, 8)
    # Flatten images into vectors(最后一维扁平化).
    dataset.data = dataset.data.reshape(dataset.data.shape[:2] + (-1,))
    # shape of dataset.data:(6877, 14, 128)
    # One-hot encode targets.
    # 将target从(6788, 14)变为(6788, 14, 26)，26为独热编码
    target = np.zeros(dataset.target.shape + (26,))
    for index, letter in np.ndenumerate(dataset.target):
        if letter:
            target[index][ord(letter) - ord('a')] = 1
    dataset.target = target
    # Shuffle order of examples.
    order = np.random.permutation(len(dataset.data))
    dataset.data = dataset.data[order]
    dataset.target = dataset.target[order]
    return dataset.data, dataset.target

if __name__ == "__main__":

    params = AttrDict(
        batch_size = 10,
        rnn_cell = tf.nn.rnn_cell.GRUCell,
        rnn_hidden_layer_size = 300, 
        optimizer = tf.train.RMSPropOptimizer(0.002),
        gradient_clipping = 5,
    )
    data_set, target = get_dataset()

    #split data in train and varify set
    split_border = int(0.8 * len(data_set))
    train_data, test_data = data_set[:split_border], data_set[split_border:]
    train_label, test_label = data_set[:split_border], data_set[split_border:]

    #data:(6877, 14, 128)
    #target:(6788, 14, 26)
    _, max_length, coding_size = data_set.shape
    one_hot_size = target.shape[2]

    #build model
    input_data = tf.placeholder(tf.float32, [None, max_length, coding_size], name="input_data")
    input_label = tf.placeholder(tf.float32, [None, max_length, one_hot_size], name="input_label")
    batch_data = batched.get_batch(data_set, target, params.batch_size)
    standard_model = rnn_module.SequenceLabelingModule(input_data, input_label, params)
    bidirection_model = bidirection_rnn_module.BidirectionSequenceLabelingModule(input_data, input_label, params)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index, data in enumerate(batch_data):
            feed_dict = {
                input_data: data[0],
                input_label: data[1],
            }
            seq_length = sess.run(standard_model.length, feed_dict)
            bi_seq_length = sess.run(bidirection_model.length, feed_dict)
            print(seq_length)
            print(bi_seq_length)  
            break     









