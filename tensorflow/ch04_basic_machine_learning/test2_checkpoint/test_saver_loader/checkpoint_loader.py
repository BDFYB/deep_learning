#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import sys

def inference(X):
    #计算推断模型在数据X上的输出，并将结果返回
    pass

def loss(X, Y):
    #计算损失函数
    return tf.constant(0)

def inputs():
    #读取或生成训练数据X极其label
    return np.array([1, 1])

def train(total_loss):
    #依据计算的总损失训练参数
    return tf.add(total_loss, 1)

def evaluate(sess, X, Y):
    #对训练的模型进行评估
    pass

#a = tf.Variable(0)
a = tf.Variable(2)
saver = tf.train.Saver()
with tf.Session() as sess:
    #saver.restore(sess, 'persist.ckpt')
    saver.restore(sess, 'test_per')
    print sess.run(a)


"""
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op  = train(total_loss)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    initial_step = 0
    data_dir = '/Users/baidu/AI/deep_learning/tensorflow/ch04_basic_machine_learning/test2_checkpoint'
    ckpt = tf.train.get_checkpoint_state(data_dir)
    print ckpt.model_checkpoint_path
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        #saver.restore(sess, 'test_per')
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        print "continue at step: %s" % initial_step
    print ("a", sess.run(a))
    
    training_steps = 500
    for step in range(initial_step, training_steps):
        sess.run([train_op])
        if step % 100 == 0:
            print "loss: ", sess.run([total_loss])
        
        if step % 300 == 0:
            saver.save(sess, 'my_module', global_step = step)
    evaluate(sess, X, Y)
    saver.save(sess, 'my_module', global_step = training_steps)
    
    coord.request_stop()
    coord.join(threads)
    sess.close()
"""
