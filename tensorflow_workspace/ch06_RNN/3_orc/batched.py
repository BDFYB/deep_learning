#encoding=utf-8
import tensorflow as tf 

class get_batch(object):
    def __init__(self, data, target, batch_size):
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.length = len(self.data)

    def __iter__(self):
        total_index = 0
        current_size = 0
        batched_data = []
        batched_target = []        

        while True:

            if current_size == 0:
                batched_data = []
                batched_target = []
            if (total_index >= self.length):
                break
            batched_data.append(self.data[total_index])
            batched_target.append(self.target[total_index])
            total_index += 1
            current_size += 1
            if current_size >= self.batch_size:
                current_size = 0
                yield batched_data, batched_target

