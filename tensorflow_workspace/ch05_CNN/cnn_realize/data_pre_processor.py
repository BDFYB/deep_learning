#encoding=utf-8

#存储一次数据（一张图片）的过程
#1、read_file()读取数据
#2、decode_jpeg()解析数据
#3、图片处理（resize/gray_scale等）
#4、类型转化为string
#5、制作example数据
#6、序列化后，write进tfrecord

# 数据大小：121个种类，20939张图片。每个种类80%用作训练，20%用作验证

import tensorflow as tf
import os
import glob
import itertools
import collections

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("image_dir","./data/Images/", "image dir name")
tf.flags.DEFINE_string("test_image_dir","./data/test_images/", "image dir name")
tf.flags.DEFINE_string("record_location","./data/processed_tf_records/", "image tf-records dir name")
tf.flags.DEFINE_string("test_record_location","./data/processed_tf_records_test_small/", "image tf-records dir name")


def _make_dataset_by_raw_data(file_name):
    #制作整理后的数据结构
    image_file_names = glob.glob(file_name)
    #print(image_file_names[0:2])
    training_data_set = collections.defaultdict(list)
    testing_data_set = collections.defaultdict(list)
    
    # map(fun, iterable):对可迭代函数'iterable'中的每一个元素应用'function'方法，将结果作为list返回
    file_name_with_label = map(lambda filename:(filename.split("/")[3], filename), image_file_names)
    for dog_label, image in itertools.groupby(file_name_with_label, lambda x:x[0]):
        #enumerate(list): 返回list的索引和元素
        for i, breed_image in enumerate(image):
            if i % 5 == 0:
                testing_data_set[dog_label].append(breed_image[1])
            else:
                training_data_set[dog_label].append(breed_image[1])

        #测试每个品种测试数据是否覆盖18%
        test_len = len(testing_data_set[dog_label])
        train_len = len(training_data_set[dog_label])
        assert round(test_len/(test_len + train_len), 2) > 0.18
    return training_data_set, testing_data_set


def _make_tfr_file(data_dict, output_dir):
    error_log = "data_processed_error.log"
    file_object = open(error_log, 'w')
    sess = tf.Session()
    count_index = 0
    file_writer = None
    for label, image_list in data_dict.items():
        for image in image_list:
            if count_index % 100 == 0:
                if file_writer:
                    file_writer.close()
                output_file_name = "{output_dir}-{count_index}.tfrecords".format(
                                    output_dir = output_dir,
                                    count_index = count_index)
                file_writer = tf.python_io.TFRecordWriter(output_file_name)
            count_index += 1
            image_data = tf.read_file(image)
            try:
                decoded_image = tf.image.decode_jpeg(image_data)
            except Exception as e:
                print("%s parse failed: %s" % (image, e))
                file_object.write(err)
                continue
            try:
                grayscale_image = tf.image.rgb_to_grayscale(decoded_image)
                resized_image = tf.image.resize_images(grayscale_image, [250, 151])
            except Exception as e:
                err = "%s parse grayscale failed: %s" % (image, e)
                file_object.write(err)

                continue

            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            image_label = label.encode("utf-8")

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))
            file_writer.write(example.SerializeToString())
    file_writer.close()
    file_object.close()
    sess.close()


if __name__ == "__main__":
    image_file_names = FLAGS.image_dir + "n02*/*.jpg"
    training_data_set, testing_data_set = _make_dataset_by_raw_data(image_file_names)
    if not os.path.exists(FLAGS.record_location):
        os.mkdir(FLAGS.record_location)

    _make_tfr_file(training_data_set, FLAGS.record_location + "train") 
    _make_tfr_file(testing_data_set, FLAGS.record_location + "test")

