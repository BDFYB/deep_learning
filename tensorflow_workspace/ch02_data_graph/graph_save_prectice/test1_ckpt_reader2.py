import tensorflow as tf   

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("checkpoint_dir","./saved_checkpoint_test1/", "save dir name")
tf.flags.DEFINE_string("checkpoint_file","test1_model.ckpt-1000.meta", "save file name")

#直接加载计算图

graph = tf.Graph()


with tf.Session(graph = graph) as sess:
    #First let's load meta graph and restore weights
    # 载入图结构，保存在.meta文件中
    saver = tf.train.import_meta_graph(FLAGS.checkpoint_dir + FLAGS.checkpoint_file)
    # 通过检查点文件锁定最新的模型
    # 载入参数，参数保存在两个文件中，不过restore会自己寻找
    saver.restore(sess,tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    # Access saved Variables directly
    print(sess.run('bias:0'))
    # This will print 2, which is the value of bias that we saved
    
    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data
    
    w1 = graph.get_tensor_by_name("w1:0")
    w2 = graph.get_tensor_by_name("w2:0")

    feed_dict ={w1:13.0,w2:17.0}

    #Now, access the op that you want to run. 
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    print(sess.run(op_to_restore, feed_dict))
    #This will print 60 which is calculated

                     