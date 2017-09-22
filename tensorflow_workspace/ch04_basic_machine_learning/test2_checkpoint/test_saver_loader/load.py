# coding=utf-8
#saver提取变量测试
import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w = tf.Variable(np.arange(9).reshape((3,3)),dtype=tf.float32)
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32)
a = tf.Variable(np.arange(4).reshape((2,2)),dtype=tf.float32)
program = []
program +=[w,b]
saver =tf.train.Saver(program)
with tf.Session() as sess:

    saver.restore(sess,'persist.ckpt')
    print ("weights",sess.run(w))
    print ("b",sess.run(b))
    #print ("s",sess.run(a))
