import tensorflow as tf
import numpy as np
import csv
from tkinter import _flatten
import sklearn

from sklearn.preprocessing import MinMaxScaler


import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib import pyplot as plt

csvFile = open(r'C:\Users\tangrong\Desktop\flow1140.csv', "r")
reader = csv.reader(csvFile)
list1=[]
list2=[]
#全部读进来，再操作
# print(reader)
for row in reader:
    #list1 += [row[1],row[2],row[3]]
    list2 += [row[0]]#它迭代的读取貌似是一个一个数据读

#list1 = np.array(list1).reshape(len(list1)//3,3)
#list2 = np.array(list2).reshape(len(list2),1)
delaytime = list(_flatten(list2))
delaytime = np.array(delaytime,dtype= np.float32)

a=delaytime.reshape(-1,1)#MinMaxScaler归一化要求函数必须为2维，因为是按照列进行归一的，
# flow = []
#Sklearn的MinMaxScaler，最简单的归一化
scaler= MinMaxScaler(feature_range=(0, 1))
flow = scaler.fit_transform(a).squeeze()
# print('flow:', flow)




# for x in delaytime:
#     x = float(x - np.min(delaytime))/(np.max(delaytime )- np.min(delaytime))
#     flow.append(x)
# print("delaytime:", flow)


HIDDEN_SIZE = 200
NUM_LAYERS = 2

TIME_STEPS = 2
TRAINING_STEPS =5000 #988-TIME_STEPS
BATCH_SIZE =30#batch 是计算一次cost的集合，数据量小的时候，使用一个batch即可，

TRAINING_EXAMPLES = 988
TEST_EXAMPLES = 152-TIME_STEPS
SAMPLE_GAP = 1





def generate_data(seq):
    x = []
    y = []
    for i in range (len(seq)-TIME_STEPS ):
        x.append([seq[i:i+TIME_STEPS ]])
        y.append([seq[i+TIME_STEPS ]])
    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)


def lstm_model(x,y,is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell (HIDDEN_SIZE)
                                        for _ in range(NUM_LAYERS)])
    outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    output = outputs[:,-1,:]#取的是final step的output

    predictions = tf.contrib.layers.fully_connected(output,1,activation_fn=None)

    if not is_training:
        return predictions ,None,None

    loss = tf.losses.mean_squared_error (labels= y,predictions = predictions )

    train_op = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),
                                               optimizer="Adagrad",learning_rate=0.1)
    return predictions ,loss,train_op

def train(sess,train_x,train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE )#当循环次数乘以batch size 大于 数据集的数量，就会报错 end of sequence ，
    # 所以，需要repeat重复使用数据，repeat().shuffle(1000),其实循环一次，取一次数据，执行一次cell，更新一次权重，就是计算一个batch
    x,y = ds.make_one_shot_iterator().get_next()
    print("trainx:", x.shape, "trainy:", y.shape)

    with tf.variable_scope ("model"):
        predictions,loss,train_op = lstm_model(x,y,True)

    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS ):
        _,l = sess.run([train_op,loss])
        if i % 50 == 0:
            print("train step: " + str(i) +",loss:" + str(l))

def run_eval(sess,test_x,test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(1 )
    x,y = ds.make_one_shot_iterator() .get_next()
    print("testx:", x.shape, "testy:", y.shape)

    with tf.variable_scope("model",reuse=True):
        prediction,_,_= lstm_model(x,[0.0],False)

    predictions = []
    labels = []
    for i in range (TEST_EXAMPLES  ):
        p,l = sess.run([prediction,y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions ).squeeze()
    labels = np.array(labels ).squeeze()
    print("predictions:", predictions.shape)
    print("labels:", labels.shape)
    rmse = np.sqrt(((predictions - labels )**2).mean(axis=0))
    print("Mean Square Error is: %f" % rmse)

    pre2 = predictions.reshape(-1, 1)#1列，-1是指不给定的
    lab2 = labels.reshape(-1, 1)
    pre = scaler.inverse_transform(pre2)
    lab = scaler.inverse_transform(lab2)

    plt.figure()
    # plt.plot(predictions ,label = 'predictions')
    # plt.plot(labels,label = 'real')
    plt.plot(pre, label='predictions')
    plt.plot(lab, label='real')
    plt.legend()
    plt.show()

# test_start = delaytime (TRAINING_EXAMPLES + TIME_STEPS )* SAMPLE_GAP
# test_end = test_start + (TEST_EXAMPLES  + TIME_STEPS )* SAMPLE_GAP
#
# train_x, train_y = generate_data(np.sin(np.linspace(0,test_start,TRAINING_EXAMPLES + TIME_STEPS,
#                                                     dtype=np.float32)))
# test_x,test_y = generate_data(np.sin(np.linspace(test_start ,test_end ,TEST_EXAMPLES  + TIME_STEPS,
#                                                  dtype=np.float32)))

train_x, train_y = generate_data(flow [0:988])
print("train_x:",train_x.shape,train_y.shape)
test_x,test_y = generate_data(flow [988:1140])
print("test_x:",test_x.shape,test_y.shape )
# [1,2]的shape值(2,)，意思是一维数组，数组中有2个元素。
#
# [[1],[2]]的shape值是(2,1)，意思是一个二维数组，每行有1个元素。所以shape为（188,2,4）三维，每行有四个元素，一个二维是2行，一共有188个二维
#
# [[1,2]]的shape值是（1，2），意思是一个二维数组，每行有2个元素。
with tf.Session() as sess:
    train (sess,train_x,train_y)
    run_eval(sess,test_x,test_y)

