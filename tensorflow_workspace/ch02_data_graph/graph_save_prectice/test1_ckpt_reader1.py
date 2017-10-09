import tensorflow as tf

#Prepare to feed input, i.e. feed_dict and placeholders
w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1 = tf.Variable(3.0,name="bias")
feed_dict = {w1:4, w2:8}

#Define a test operation that we will restore
w3 = tf.add(w1, w2)
w4 = tf.multiply(w3, b1,name="op_to_restore")

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #"""
    ckpt = tf.train.get_checkpoint_state('./test1_checkpoint/')
    #saver.restore(sess, ckpt.model_checkpoint_path) 
    saver.restore(sess, './test1_checkpoint/test1_model.ckpt-1000')
    #"""

    #Run the operation by feeding input
    print(sess.run(w4,feed_dict))
    #Prints 24 which is sum of (w1+w2)*b1 
