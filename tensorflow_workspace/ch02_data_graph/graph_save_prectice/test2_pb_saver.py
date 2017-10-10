import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("checkpoint_dir","./saved_pb_test2/", "save dir name")
tf.flags.DEFINE_string("checkpoint_pb_file","test_pb.pb", "save file name")
tf.flags.DEFINE_string("checkpoint_file","checkpoint.data", "save file name")

cal_graph = tf.Graph()
with cal_graph.as_default():
    input1 = tf.placeholder(tf.int32, [10], name="input")
    data = np.arange(10, dtype=np.int32)
    output1= tf.add(input1, tf.constant(100, dtype=tf.int32), name="output") #  data depends on the input data
    saved_result = tf.Variable(data, name="saved_result")
    #计算完成后将值赋给一个变量，之后方便直接从graph读取
    do_save = tf.assign(saved_result, output1)

with tf.Session(graph = cal_graph) as sess:
    #save graph
    tf.train.write_graph(sess.graph_def, FLAGS.checkpoint_dir, FLAGS.checkpoint_pb_file, False)
    print(sess.run(output1, {input1:data}))
    
    # now set the data: same to test1
    result,_ = sess.run([output1,do_save], {input1: data}) # calculate output1 and assign to 'saved_result'
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, FLAGS.checkpoint_dir + FLAGS.checkpoint_file)
