# coding=utf-8
# saver保存变量测试
import os                             
import tensorflow as tf
import numpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #有些指令集没有装，加这个不显示那些警告
w = tf.Variable([[1,2,3],[2,3,4],[6,7,8]],dtype=tf.float32)
b = tf.Variable([[4,5,6]],dtype=tf.float32,)
s = tf.Variable([[2, 5],[5, 6]], dtype=tf.float32)
init = tf.initialize_all_variables()
program = []
program += [w, b]
saver =tf.train.Saver(program)
with tf.Session() as sess:
    sess.run(init)
    sess.run(b.assign_add([[3, 2, 1]]))
    save_path = saver.save(sess, "persist.ckpt")#路径可以自己定
    print("save to path:",save_path)
