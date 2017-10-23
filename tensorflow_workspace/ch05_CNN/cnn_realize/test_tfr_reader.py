#encoding=utf-8

#读取一张图片的过程
#reader read()
#parse_single_example解析example协议
#decode_raw()bin文件数据解析
#reshape & cast
#shuffle_batch读取数据

import tensorflow as tf
import os
import glob
import itertools
import collections

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("record_location","./data/processed_tf_records/", "image tf-records dir name")
tf.flags.DEFINE_string("test_record_location","./data/processed_tf_records_test_small/", "image tf-records dir name")


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
        batch_size = 3
        capacity = min_after_dequeue + 3 * batch_size
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch


if __name__ == "__main__":
    tfr_file_names = FLAGS.test_record_location + "*.tfrecords"
    image_file_names = glob.glob(tfr_file_names)
    graph = tf.Graph()
    image_batch, label_batch = _read_records(image_file_names, graph)

    with tf.Session(graph = graph) as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(500):
            (image_batchs, label_batchs) = sess.run([image_batch, label_batch])

            print(label_batchs)

        coord.request_stop()
        coord.join(threads)
        sess.close()



