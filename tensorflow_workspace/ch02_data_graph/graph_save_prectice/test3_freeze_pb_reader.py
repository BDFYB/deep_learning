import tensorflow as tf   
import os
from tensorflow.python.framework import graph_util

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("pb_file", "./saved_freeze_pb_test3/freeze_module.pb", "save dir name")


# 直接加载计算图
# parse the graph_def file

with tf.gfile.GFile(FLAGS.pb_file, "rb") as f:  
    graph_def = tf.GraphDef()  
    graph_def.ParseFromString(f.read()) 

# load the graph_def in the default graph

graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(  
        graph_def,   
        input_map = None,   
        return_elements = None,   
        name = "old_graph",   
        op_dict = None,   
        producer_op_list = None  
    )

# We can list operations  
# op.values() gives you a list of tensors it produces  
# op.name gives you the name  
# 输入,输出结点也是operation, 所以, 我们可以得到operation的名字

for op in graph.get_operations():  
    print(op.name,op.values())  
    """
    old_graph/w1 (<tf.Tensor 'old_graph/w1:0' shape=<unknown> dtype=float32>,)
    old_graph/w2 (<tf.Tensor 'old_graph/w2:0' shape=<unknown> dtype=float32>,)
    old_graph/bias (<tf.Tensor 'old_graph/bias:0' shape=() dtype=float32>,)
    old_graph/bias/read (<tf.Tensor 'old_graph/bias/read:0' shape=() dtype=float32>,)
    old_graph/Add (<tf.Tensor 'old_graph/Add:0' shape=<unknown> dtype=float32>,)
    old_graph/op_to_restore (<tf.Tensor 'old_graph/op_to_restore:0' shape=<unknown> dtype=float32>,)
    """
    #为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字  
    #注意old_graph/op_to_restore仅仅是操作的名字, old_graph/op_to_restore:0才是tensor的名字  
    out = graph.get_tensor_by_name('old_graph/op_to_restore:0') 
    w1 = graph.get_tensor_by_name('old_graph/w1:0')
    w2 = graph.get_tensor_by_name('old_graph/w2:0') 
      
with tf.Session(graph=graph) as sess:  
    y_out = sess.run(out, feed_dict={w1:4, w2:8})  
    print(y_out) # [[ 0.]] Yay!  
    print ("finish")
  

                     