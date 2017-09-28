import tensorflow as tf

if __name__ == "__main__":
    image_file = ["./image/image1.jpg"]
    image_tfr_file = "./output/image1.tfr"
    #file_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_file))
    file_queue = tf.train.string_input_producer(image_file)
    image_reader = tf.WholeFileReader()
    _, image = image_reader.read(file_queue)
    data = tf.image.decode_jpeg(image)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        image_mat = sess.run(data)
        #print(image_mat)
        print(sess.run(tf.shape(image_mat)))

        image_label = b'\x01'
        image_hight, image_width, image_channels = image_mat.shape
        image_bytes = image_mat.tobytes()

        #TFRecord
        writer = tf.python_io.TFRecordWriter(image_tfr_file)
        #example协议
        example = tf.train.Example(features = tf.train.Features(feature={
                "label": tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_label])),
                "image": tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_bytes])),}))


        #存储tensorboard，看下读取的数据图是什么样子的(使用Tensorboard打印图片)
        graph_writer = tf.summary.FileWriter('./tfr_writer_graph', sess.graph)
        image = tf.expand_dims(image_mat, 0)
        summary_op = tf.summary.image("image1", image)

        # 运行并写入日志
        summary = sess.run(summary_op)
        graph_writer.add_summary(summary)

        graph_writer.close()
        writer.write(example.SerializeToString())
        writer.close()
        coord.request_stop()
        coord.join(threads)
        sess.close()
