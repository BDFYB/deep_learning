import tensorflow as tf   
import os
from tensorflow.python.framework import graph_util

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("checkpoint_dir", "./saved_checkpoint_test1/", "save dir name")
tf.flags.DEFINE_string("checkpoint_meta_file", "test1_model.ckpt-1000.meta", "save file name")
tf.flags.DEFINE_string("checkpoint_data_file", "test1_model.ckpt-1000", "save file name")
tf.flags.DEFINE_string("output_dir", "./saved_freeze_pb_test3/", "output dir name")
tf.flags.DEFINE_string("output_file", "freeze_module.pb", "output file name")


if os.path.exists(FLAGS.output_dir) is False:
    os.makedirs(FLAGS.output_dir)


# Before exporting our graph, we need to precise what is our output node  
# this variables is plural, because you can have multiple output nodes  
#freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点  
#输出结点可以看我们模型的定义  
#只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃。所以,output_node_names必须根据不同的网络进行修改


graph = tf.Graph()
output_node_names = ["op_to_restore"]

with tf.Session(graph = graph) as sess:
    # First let's load meta graph and restore weights
    # 载入图结构，保存在.meta文件中
    # We import the meta graph and retrive a Saver
    saver = tf.train.import_meta_graph(FLAGS.checkpoint_dir + FLAGS.checkpoint_meta_file)
    
    # We retrieve the protobuf graph definition
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights  
    # 这边已经将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen  
    # 相当于将参数已经固化在了图当中

    saver.restore(sess, FLAGS.checkpoint_dir + FLAGS.checkpoint_data_file)

    # We use a built-in TF helper to export variables to constant 
    output_graph_def = graph_util.convert_variables_to_constants(  
     sess,   
     input_graph_def,   
     output_node_names, # We split on comma for convenience  
    )   

    # Finally we serialize and dump the output graph to the filesystem  
    with tf.gfile.GFile(FLAGS.output_dir + FLAGS.output_file, "wb") as f:  
        f.write(output_graph_def.SerializeToString())  
        print("%d ops in the final graph." % len(output_graph_def.node))  

                     