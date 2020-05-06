import os
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from tfrecord import create_record


num_samples = 3000
filename = r"D:\ZSNJAP01\flight\clf\train.tfrecords"
def read_and_decode(filename, is_batch):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = {
        'img_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64) }# 将label和img数据读出来
    feature = tf.parse_single_example(serialized_example,features)
    img = tf.decode_raw(feature['img_raw'], tf.uint8)
    img = tf.reshape(img, [20])
    img = tf.cast(img, tf.float32) * (1. / 255) #- 0.5#图像减去均值处理
    # label = tf.cast(features['label'], tf.int32)#tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换
    label = feature['label']

    if is_batch:
        batch_size = 50
        min_after_dequeue = 20
        num_threads = 3
        #capacity = (min_after_dequeue + (num_threads + a small safety margin)∗batchsize)
        capacity = min_after_dequeue + (num_threads + 1) * batch_size
        img, label = tf.train.shuffle_batch([img, label],
                                            batch_size=batch_size,
                                            num_threads=num_threads,
                                            capacity=capacity,
                                            min_after_dequeue=min_after_dequeue)
        #生成的batch得是2维的
    return img, label


init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    minist = read_and_decode(filename, is_batch=True)
    #开启一个协调器,,这样可以在发生错误的情况下正确地关闭这些线程
    coord = tf.train.Coordinator()
    #使用start_queue_runners 启动队列填充,
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    train_img,train_label = read_and_decode("train.tfrecords", is_batch=False)

    for i in range(num_samples):
        example, lab = sess.run([train_img, train_label])  # 在会话中取出image和label
        #img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
        # img.save(gen_picture + '/' + str(i) + 'samples' + str(lab) + '.jpg')  # 存下图片;注意cwd后边加上‘/’
        # img.save( '/' + str(i) + 'samples' + str(lab) + '.jpg')  # 存下图片;注意cwd后边加上‘/’
        print(example.shape, lab)
    coord.request_stop()
    coord.join(threads)
    sess.close()  # 关闭会话

    try:
        while not coord.should_stop():
            print('************')
            # 获取每一个batch中batch_size个样本和标签
            image, label = sess.run([train_img,train_label])
            print(image.shape, label)
    except tf.errors.OutOfRangeError:  # 如果读取到文件队列末尾会抛出此异常
        print("done! now lets kill all the threads……")
    finally:
        # 协调器coord发出所有线程终止信号
        coord.request_stop()
        print('all threads are asked to stop!')
    coord.join(threads)  # 把开启的线程加入主线程，等待threads结束
    print('all threads are stopped!')