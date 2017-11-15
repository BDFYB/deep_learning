import tensorflow as tf 
import os
import data_processor
import batched
import numpy as np


def get_dataset():
    dataset = data_processor.OcrData(use_local_file = True)
    # shape of dataset.data:(6877, 14, 16, 8)
    # Flatten images into vectors(最后一维扁平化).
    dataset.data = dataset.data.reshape(dataset.data.shape[:2] + (-1,))
    # shape of dataset.data:(6877, 14, 128)
    # One-hot encode targets.
    # 将target从(6788, 14)变为(6788, 14, 26)，26为独热编码
    target = np.zeros(dataset.target.shape + (26,))
    for index, letter in np.ndenumerate(dataset.target):
        if letter:
            target[index][ord(letter) - ord('a')] = 1
    dataset.target = target
    # Shuffle order of examples.
    order = np.random.permutation(len(dataset.data))
    dataset.data = dataset.data[order]
    dataset.target = dataset.target[order]
    return dataset.data, dataset.target


if __name__ == "__main__":
    #是否使用双向RNN
    IS_BIDIRECTION = True

    if IS_BIDIRECTION:
        FORZEN_GRAPH = "./frozen_graph/forzen_bidirection_module.pb" 
    else:
        FORZEN_GRAPH = "./frozen_graph/forzen_module.pb"
        
    if not os.path.isfile(FORZEN_GRAPH):
        raise("frozen graph file not exsists")

    with tf.gfile.GFile(FORZEN_GRAPH, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read()) 
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(  
            graph_def,   
            input_map = None,   
            return_elements = None,   
            name = "frozen_module",   
            op_dict = None,   
            producer_op_list = None  
        )
        input_data_holder = graph.get_tensor_by_name("frozen_module/input_data:0")
        input_label_holder = graph.get_tensor_by_name("frozen_module/input_label:0")
        prediction = graph.get_tensor_by_name("frozen_module/prediction:0")

        """
        #查看计算图所有op 
        for op in graph.get_operations():  
            print(op.name,op.values()) 
        """    

    data, target = get_dataset()
    border = int(0.8 * len(data))
    test_data = data[border:]
    test_label = target[border:]

    batch_data = batched.get_batch(test_data, test_label, 1)
    
    with tf.Session(graph=graph) as sess:

        for index, batch_data in enumerate(batch_data):
            feed_dict = {
                input_data_holder: batch_data[0],
                input_label_holder: batch_data[1],
            }

            result = sess.run(prediction, feed_dict)
            #print(np.array(result).shape)
            predict_string = ""
            for letter in result[0]:
                max_index = letter.tolist().index(max(letter))
                if max_index == 0:
                    break
                a = chr(max_index+ord('a'))
                predict_string = predict_string + a
            print(predict_string)
            #真实单词
            real_label = batch_data[1]
            real_string = ""
            for letter in real_label[0]:
                max_index = letter.tolist().index(max(letter))
                if max_index == 0:
                    break
                a = chr(max_index+ord('a'))
                real_string = real_string + a
            print(real_string)
            










