from __future__ import print_function
import tensorflow as tf
from PIL import Image
import  numpy as np
import os
import pickle


def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
    #im.show()
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    new_data = np.reshape(data,(width,height))
    return new_data


imgary = []
root = r'D:\ZSNJAP01\flight\imagetime'+ '\\' + 'img_'

for i in range(3000):
    data = ImageToMatrix(root + str(i)+ '.jpg')
    imgary.append(data)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#产生随机变量
    return initial

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return initial

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def get_feature(array):
    xs = array
    x_image= tf.reshape(xs,[-1,5,4,1])

    W_conv1= weight_variable([5,5,1,32])
    b_conv1= bias_variable([32])
    h_conv1=tf.nn.relu(conv2d(x_image ,W_conv1 )+b_conv1)#conv1 output,stride =1 ,padding is 0,output_size = [5,4,32]

    h_pool1=max_pool_2x2(h_conv1 )#pool1 outout output_size = [3,2,32]

    W_conv2= weight_variable([5,5,32,64]) #patch 5*5（范围）,in size 32（输入厚度）,out size 64(输出后厚度)
    b_conv2= bias_variable([64])
    h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2 )+b_conv2)#output size 3*2*64
    h_pool2=max_pool_2x2(h_conv2) #output size 2*1*64 这里的h_pool2 就是特征

    return h_pool2






imgaryf = np.zeros(((len(imgary),len(imgary[0]),len(imgary[0][0]))),dtype= 'float32')#存放图片的像素
featime = []
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for k in range(len(imgary)):
    for i in range(len(imgary[k])):
        for j in range(len(imgary[k][i])):
            imgaryf[k,i,j] = float(imgary[k][i,j])
    feature = get_feature(imgaryf[k])#[1,2,1,64]tensor
    a = feature.eval(session=sess)#[1,2,1,64]ndarray
    a = np.squeeze(a,axis = 0)#[2,1,64]
    a = a.tolist()#[2,1,64]
    print(a)

    featime.append(a)#如果a是一个array，然后featuretime是一个list，会造成不兼容

b = os.path.exists(r"D:\ZSNJAP01\flight\feature")
if b :
    print("path exist")
else:
    os.makedirs(r'D:\ZSNJAP01\flight\feature')

with open(r"D:\ZSNJAP01\flight\feature\featime.pkl",'wb') as f:
    pickle.dump(featime, f)



