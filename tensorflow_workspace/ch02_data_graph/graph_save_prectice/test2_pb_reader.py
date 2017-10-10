import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("checkpoint_dir","./saved_pb_test2/", "save dir name")
tf.flags.DEFINE_string("checkpoint_pb_file","test_pb.pb", "save file name")
tf.flags.DEFINE_string("checkpoint_file","checkpoint.data", "save file name")

g1 = tf.Graph()
with g1.as_default():
    with gfile.FastGFile(FLAGS.checkpoint_dir + FLAGS.checkpoint_pb_file, 'rb') as fd:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fd.read())
        tf.import_graph_def(graph_def, name='')
        
        input_x = g1.get_tensor_by_name("input:0")
        output = g1.get_operation_by_name("output")


with tf.Session(graph=g1) as sess:
    """
    input_x = sess.graph.get_tensor_by_name("input:0")
    print(input_x)
    Const = sess.graph.get_tensor_by_name("Const:0")
    print(Const)
    output = sess.graph.get_operation_by_name("output")
    print(output)
    """
    print(input_x)
    print(output)
    data1 = np.arange(10)+100
    print("data1:", data1)
    data = np.arange(10)

    persisted_result = sess.graph.get_tensor_by_name("saved_result:0")
    tf.add_to_collection(tf.GraphKeys.VARIABLES, persisted_result)
    try:
        saver = tf.train.Saver(tf.all_variables()) # 'Saver' misnomer! Better: Persister!
    except:
        pass
    print("load data")
    saver.restore(sess, FLAGS.checkpoint_dir + FLAGS.checkpoint_file)  # now OK

    #restore之后，我们的变量才被初始化
    print(persisted_result.eval())
    print("DONE")

    #print None, this method can only save graph, not data.
    print(sess.run(output, {input_x:data}))
    #print("result:",result)
