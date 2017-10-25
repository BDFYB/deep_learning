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


def _load_module_graph(graph):
    with tf.gfile.GFile(FLAGS.trained_module, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read()) 

    # load the graph_def in the default graph


    with graph.as_default():
        tf.import_graph_def(  
            graph_def,   
            input_map = None,   
            return_elements = None,   
            name = "cnn",   
            op_dict = None,   
            producer_op_list = None  
        )
    """
    for op in graph.get_operations():  
        print(op.name,op.values()) 
    exit(0)
    """

def _load_single_data(graph, data_dict, total_labels):
    with tf.Session(graph=graph) as sess:
        label = ''
        file_name = ''
        for file, label in data_dict.items():
            file_name = file
            label = label
        if not label in labels:
            print("label: %s not in labels" % file_name)
            return [], []
        label_batch_in_index = labels.index(label)

        if not os.path.exists(file_name):
            print("%s not exists" % file_name)
            return [], []

        image_data = tf.read_file(file_name)
        try:
            decoded_image = tf.image.decode_jpeg(image_data)
        except Exception as e:
            print("%s parse failed: %s" % (file_name, e))
            return [], []
        try:
            grayscale_image = tf.image.rgb_to_grayscale(decoded_image)
            resized_image = tf.image.resize_images(grayscale_image, [250, 151])
        except Exception as e:
            err = "%s parse grayscale failed: %s" % (file_name, e)
            return [], []
        image_batch_data =[]
        image_batch_data.append(sess.run(resized_image))
        label_batch_data = []
        label_batch_data.append(label_batch_in_index)   
        return image_batch_data, label_batch_data


if __name__ == "__main__":
    # 1. get varify data
    pic_dict = _get_pic_dict(FLAGS.verify_pic_location)
    #for k in pic_dict:
    #    print(k)

    # 2. get frozen mudule
    if not os.path.exists(FLAGS.trained_module):
        print("module file not exists!")
        exit(0)
    graph = tf.Graph()
    _load_module_graph(graph)

    # 3. Find every directory name in the imagenet-dogs directory (n02085620-Chihuahua, ...)
    labels = list(map(lambda c: c.split("/")[-1], glob.glob("./data/Images/*")))
    with tf.Session(graph=graph) as sess:
        for data_dict in pic_dict:
            # 4. load pict data
            image_batch_data, label_batch_data = _load_single_data(graph, data_dict, labels)
            if len(image_batch_data) == 0:
                continue
            input_data_batch = graph.get_tensor_by_name('cnn/feed_image_batch:0')
            input_label_batch = graph.get_tensor_by_name('cnn/feed_train_labels:0')
            pridict_op = graph.get_tensor_by_name('cnn/infer:0')
            feed_dict = {
                input_data_batch: image_batch_data,
                input_label_batch: label_batch_data,
            }
            result_mat = sess.run(pridict_op, feed_dict)
            print(label_batch_data)
            print(result_mat[0])
