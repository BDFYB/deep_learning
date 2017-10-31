import time

class Processor(object):
    def __init__(self):
        pass

    def __iter__(self):
        for i in range(10):
            print("class outer loop start")
            for j in range(10, 20):
                print("class inner loop start")
                yield i+j
                print("class inner loop end")
            print("class outer loop end")
            yield i


if __name__ == "__main__":
    iterable_obj = Processor()
    for i in iterable_obj:
        print("new in main loop")
        print(i)
        time.sleep(2)