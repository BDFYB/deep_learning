#encoding=utf-8
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    image_hight = 939
    image_width = 626
    channel = 3

    image_file = ["./image/image1.jpg"]
    image_tfr_file = ["./output/image1.tfr"]
    #file_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_file))
    file_queue = tf.train.string_input_producer(image_tfr_file)
    record_reader = tf.TFRecordReader()
    _, serialized_record = record_reader.read(file_queue)

    tf_record_features = tf.parse_single_example(serialized_record,
                                                 features = {
                                                    "label": tf.FixedLenFeature([], tf.string),
                                                    "image": tf.FixedLenFeature([], tf.string),
                                                 })

    image_mat_record = tf.decode_raw(tf_record_features["image"], tf.uint8)
    #这样读出的是一个List，需要对image_mat进行reshape
    image_mat = tf.reshape(image_mat_record, [1, image_hight, image_width, channel])
    image_mat = tf.cast(image_mat, tf.float32)

    image_label = tf.cast(tf_record_features["label"], tf.string)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        (image_mat, image_label) = sess.run((image_mat, image_label))
        #print(image_mat)
        print(image_label)
        print(sess.run(tf.shape(image_mat)))

        #对图像进行下处理，为了查看图像。正常情况下处理要放到存储部分，处理后直接存为TFRecord文件
        kernel = tf.constant([
            [[[-1.], [0.0], [0.0]], [[0.0], [-1.], [0.0]], [[0.0], [0.0], [-1.]]],
            [[[-1.], [0.0], [0.0]], [[0.0], [-1.], [0.0]], [[0.0], [0.0], [-1.]]],
            [[[-1.], [0.0], [0.0]], [[0.0], [-1.], [0.0]], [[0.0], [0.0], [-1.]]],
        ])
        #需要四维输入
        image_mat_out = sess.run(tf.nn.conv2d(image_mat, kernel, strides = [1, 1, 1, 1], padding='VALID'))
        print(sess.run(tf.shape(image_mat_out)))
        #看下读取的数据图是什么样子的(使用Tensorboard打印图片)
        graph_writer = tf.summary.FileWriter('./tensorboard_tfr_reader_graph', sess.graph)

        # image = tf.expand_dims(image_mat_out, 0)
        summary_op = tf.summary.image("image1", image_mat_out)

        # 运行并写入日志
        summary = sess.run(summary_op)
        graph_writer.add_summary(summary)

        graph_writer.close()

        coord.request_stop()
        coord.join(threads)
        sess.close()
