import tensorflow as tf
#只加载变量
FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("checkpoint_dir","./saved_checkpoint_test1/", "save dir name")
tf.flags.DEFINE_string("checkpoint_file","test1_model.ckpt-1000", "save file name")

#Prepare to feed input, i.e. feed_dict and placeholders
w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1 = tf.Variable(3.0,name="bias")
feed_dict = {w1:4, w2:8}

#Define a test operation that we will restore
w3 = tf.add(w1, w2)
w4 = tf.multiply(w3, b1, name="op_to_restore")

# Add ops to save and restore all the variables.
# 程序前面得有 Variable 供 save or restore 才不报错
# 否则会提示没有可保存的变量
saver = tf.train.Saver()

with tf.Session() as sess:
    #restore则不需初始化，使用的是持久化的变量数据
    #sess.run(tf.global_variables_initializer())

    saver.restore(sess, FLAGS.checkpoint_dir + FLAGS.checkpoint_file)


    #Run the operation by feeding input
    print(sess.run(w4,feed_dict))
    #Prints 24 which is sum of (w1+w2)*b1 
