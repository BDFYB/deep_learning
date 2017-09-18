# coding=utf-8
import os
import tensorflow as tf
import numpy
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #有些指令集没有装，加这个不显示那些警告
a = tf.Variable(0)
init = tf.initialize_all_variables()
saver =tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    sess.run(a.assign_add(2))
    print sess.run(a)
    save_path = saver.save(sess, "persist.ckpt")#路径可以自己定
    print("save to path:",save_path)
