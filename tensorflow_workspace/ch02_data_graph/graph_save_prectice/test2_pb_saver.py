import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile

input1= tf.placeholder(tf.int32, [10], name="input")
data = np.arange(10)
output1= tf.add(input1, tf.constant(1), name="output")

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, "test2_pb", "test_pb.pb", False)
    print(sess.run(output1,{input1:data}))
