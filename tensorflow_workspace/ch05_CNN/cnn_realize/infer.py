#encoding=utf-8

import tensorflow as tf
import os
import glob

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("verify_pic_location","./data/verify_images/n02*/*.jpg", "image tf-records dir name")
# 数据大小：121个种类，20939张图片。每个种类80%用作训练(16751)，20%用作验证
tf.flags.DEFINE_string("test_rounds", 4000, "test data")
# result module
tf.flags.DEFINE_string("trained_module", "./trained_module/frozen_inference_graph.pb", "result dir")


def _get_pic_dict(file_name):
    image_file_names = glob.glob(file_name)
    #print(image_file_names[0:2])
    return map(lambda x: {x: x.split('/')[3]}, image_file_names)


def _get_main_module():
    pass



def _read_records(file_name, graph):
    with graph.as_default():
        #filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(file_name))
        filename_queue = tf.train.string_input_producer(file_name)
        reader = tf.TFRecordReader()
        _, serialized = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized,
            features={
                'label': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string),
            })

        record_image = tf.decode_raw(features['image'], tf.uint8)

        # Changing the image into this shape helps train and visualize the output by converting it to
        # be organized like an image.
        image = tf.reshape(record_image, [250, 151, 1])

        label = tf.cast(features['label'], tf.string)

        min_after_dequeue = 10
        batch_size = FLAGS.batch_size
        capacity = min_after_dequeue + 3 * batch_size
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    return image_batch, label_batch


def cnn_train(image_batch, train_labels, graph):
    with graph.as_default():
        float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)
        conv2d_layer_one = tf.contrib.layers.convolution2d(
            float_image_batch,
            num_outputs=32,     # The number of filters to generate, 即有多少个卷积核
            kernel_size=(5,5),          # It's only the filter height and width. 卷积核数量
            activation_fn=tf.nn.relu,   # 激活函数
            weights_initializer=tf.random_normal_initializer(stddev=0.005),
            stride=(2, 2),              # 卷积步长
            trainable=True)

        pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')


        # Note, the first and last dimension of the convolution output hasn't changed but the
        # middle two dimensions have.
        conv2d_layer_one.get_shape(), pool_layer_one.get_shape()

        conv2d_layer_two = tf.contrib.layers.convolution2d(
            pool_layer_one,
            num_outputs=64,        # More output channels means an increase in the number of filters
            kernel_size=(5,5),
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer(stddev=0.005),
            stride=(1, 1),
            trainable=True)
        pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')

        conv2d_layer_two.get_shape(), pool_layer_two.get_shape()

        flattened_layer_two = tf.reshape(
            pool_layer_two,
            [
                FLAGS.batch_size,  # Each image in the image_batch
                -1           # Every other dimension of the input
            ])

        flattened_layer_two.get_shape()

        # The weights_initializer parameter can also accept a callable, a lambda is used here  returning a truncated normal
        # with a stddev specified.
        hidden_layer_three = tf.contrib.layers.fully_connected(
            flattened_layer_two,
            512,
            #todo find out what does this mean
            #weights_initializer=lambda i, dtype: tf.truncated_normal([38912, 512], stddev=0.1),
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            activation_fn=tf.nn.relu
        )

        # Dropout some of the neurons, reducing their importance in the model
        hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)

        # The output of this are all the connections between the previous layers and the 120 different dog breeds
        # available to train on.
        final_fully_connected = tf.contrib.layers.fully_connected(
            hidden_layer_three,
            120,  # Number of dog breeds in the ImageNet Dogs dataset
            #weights_initializer=lambda i, dtype: tf.truncated_normal([512, 120], stddev=0.1)
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )

        # start training
        # setup-only-ignore
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=final_fully_connected, labels=train_labels))

        # 一定把trainable=False，否则参与训练
        g_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            0.001,
            g_steps,
            120,
            0.95,
            staircase=True)

        optimizer = tf.train.AdamOptimizer(
            learning_rate, 0.9).minimize(
            loss, global_step=g_steps)

        #infer
        train_prediction = tf.nn.softmax(final_fully_connected, name="infer")

    return loss, optimizer, train_prediction

if __name__ == "__main__":
    # 1、get varify data
    pic_dict = _get_pic_dict(FLAGS.verify_pic_location)
    #for k in pic_dict:
    #    print(k)

    #2、get frozen mudule
    if not os.path.exists(FLAGS.trained_module):
        print("module file not exists!")
        exit(0)

    with tf.gfile.GFile(FLAGS.trained_module, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read()) 

    # load the graph_def in the default graph

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(  
            graph_def,   
            input_map = None,   
            return_elements = None,   
            name = "cnn",   
            op_dict = None,   
            producer_op_list = None  
        )

    # Find every directory name in the imagenet-dogs directory (n02085620-Chihuahua, ...)
    labels = list(map(lambda c: c.split("/")[-1], glob.glob("./data/Images/*")))
    sess = tf.Session(graph=graph)
    for data_dict in pic_dict:
        pridict_op = graph.get_tensor_by_name('cnn/infer:0')
        print(sess.run(pridict_op))
        break

    """
    # 1.build data input
    tfr_file_names = FLAGS.record_location + "train*.tfrecords"
    rounds = FLAGS.train_rounds
    if FLAGS.test_mode:
        print("Running test training mode")
        tfr_file_names = FLAGS.test_record_location + "train*.tfrecords"
        rounds = FLAGS.test_train_rounds

    image_file_names = glob.glob(tfr_file_names)
    graph = tf.Graph()
    image_batch, label_batch = _read_records(image_file_names, graph)

    # Match every label from label_batch and return the index where they exist in the list of classes
    train_labels = tf.map_fn(lambda l: tf.where(tf.equal(labels, l))[0,0:1][0], label_batch, dtype=tf.int64)

    # 2.build graph
    loss, optimizer, train_prediction = cnn_train(image_batch, train_labels, graph)
    output_node_names = ["infer"]
    # 3 tensorboard 

    loss_summary = tf.summary.scalar("train_loss", loss)
    summary_op = tf.summary.merge([loss_summary])
    #total_summary_op = tf.summary.merge_all()

    # 4.start training
    with tf.Session(graph = graph) as sess:
        input_graph_def = graph.as_graph_def()
        train_summary_writer = tf.summary.FileWriter(FLAGS.train_summary_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for current_step in range(rounds):
            if current_step % 500 == 0:
                print("step: %s" % current_step)
                train_summary_writer.flush()

            _, g_loss, g_summary, t_labels = sess.run(
                        [optimizer, loss, summary_op, train_labels])
            #print(t_labels)
            #print(g_loss)

            train_summary_writer.add_summary(g_summary, current_step)

        #make freeze module for infer
        output_graph_def = graph_util.convert_variables_to_constants(  
         sess,   
         input_graph_def,   
         output_node_names, # We split on comma for convenience  
        ) 

        # Finally we serialize and dump the output graph to the filesystem  
        with tf.gfile.GFile(FLAGS.trained_module, "wb") as f:  
            f.write(output_graph_def.SerializeToString())  
            print("%d ops in the final graph." % len(output_graph_def.node))  

        coord.request_stop()
        coord.join(threads)
        sess.close()
        """
