import os
import tensorflow as tf
from PIL import Image
import numpy as np
# 图片制作TFRecords数据
def create_record(classes,inpath,outpath,volumn):
    writer = tf.python_io.TFRecordWriter(outpath + '\\'+ volumn + '.tfrecords')
    for index, name in enumerate(classes): # 遍历，获取对象下标和对象值
        labels = np.zeros(len(classes),dtype = 'int64')
        class_path = inpath + '\\' + name + '\\' #通过name文件名去一个一个类添加标签
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            imgsize = img.resize((1,img.width * img.height))  # 设置需要转换的图片大小
            labels[index] = 1 #只有这个类对应的位置为1
            ###图片灰度化######################################################################
            # img=img.convert("L")
            ##############################################################################################
            img_raw = imgsize.tobytes()  # 将图片转化为原生bytes
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])), # 一张图片的像素放在一行，比如5*4
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value = labels)),#label就是类，只有这个类位置为1，其余都是0
                }))#value = [value],括号里必须是整数，不能是list，array，所以直接写labels
            writer.write(example.SerializeToString())
    writer.close()


# class的存储位置
inpath = r'D:\ZSNJAP01\flight\class'
# 生成tf的存储位置
b = os.path.exists(r'D:\ZSNJAP01\flight\clf')
if b:
    print("path exist")
else:
    os.makedirs(r'D:\ZSNJAP01\flight\clf')
# outpath = os.getcwd() + r'D:\ZSNJAP01\flight\clf'
outpath =  r'D:\ZSNJAP01\flight\clf'
classes = ('0','1')
volumn = 'train'
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    tfrecord = create_record(classes,inpath, outpath, volumn)

# 一维数组制作tf.record






