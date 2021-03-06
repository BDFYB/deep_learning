import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("checkpoint_dir","./saved_checkpoint_test1/", "save dir name")
tf.flags.DEFINE_string("checkpoint_file","test1_model.ckpt", "save file name")

#Prepare to feed input, i.e. feed_dict and placeholders
w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1 = tf.Variable(2.0, name="bias")
feed_dict = {w1:4, w2:8}

#Define a test operation that we will restore
w3 = tf.add(w1, w2)
w4 = tf.multiply(w3, b1, name="op_to_restore")

#Create a saver object which will save all the variables
saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    #Run the operation by feeding input
    print(sess.run(w4, feed_dict))
    #Prints 24 which is sum of (w1+w2)*b1 

    #Now, save the graph
    saver.save(sess, FLAGS.checkpoint_dir + FLAGS.checkpoint_file, global_step=1000)
