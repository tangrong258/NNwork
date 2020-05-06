import tensorflow as tf
import numpy as np
import csv
from tkinter import _flatten
from sklearn import preprocessing#这个函数实现对多维数组的归一化
import sklearn

from sklearn.preprocessing import MinMaxScaler

import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib import pyplot as plt

HIDDEN_SIZE = 200
NUM_LAYERS = 2

TIME_STEPS = 6
TRAINING_STEPS =5000 #988-TIME_STEPS
BATCH_SIZE =38-TIME_STEPS#batch 是计算一次cost的集合，数据量小的时候，使用一个batch即可，

TRAINING_EXAMPLES = (38-TIME_STEPS)*26
TEST_EXAMPLES = (38-TIME_STEPS)*4
SAMPLE_GAP = 1


#读入数据的格式保持时间段*天数，然后先处理每一天的数据，再合为一个序列去训练神经网路

my_matrix = np.loadtxt(open(r"C:\Users\tangrong\Desktop\flow3831.csv","rb"),delimiter=",",skiprows=0)
flow = np.array(my_matrix,dtype= np.float32)
# flow = preprocessing.scale(flow)

flow = flow.T
m = flow.shape[0]
n = flow.shape[1]
print("flow:",flow.shape)
print ("flow[1]",flow[1].shape)
train_x = np.zeros(shape=(1,TIME_STEPS))
train_y = np.zeros(shape=(1,1))
test_x = np.zeros(shape=(1,TIME_STEPS))
test_y = np.zeros(shape=(1,1))
# flowpro = np.zeros(shape=(26,32,6))#定义空的数组



def maxminnorm(array):
    # maxcols=array.max(axis=0)
    max = np.max(array)
    # mincols=array.min(axis=0)
    min = np.min(array)
    print(max,min)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        # t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
        t[:, i] = (array[:, i] - min) / (max - min)
    return t



def generate_data(seq):
    x = []
    y = []
    for i in range (len(seq)-TIME_STEPS ):
        x.append([seq[i:i+TIME_STEPS ]])
        y.append([seq[i+TIME_STEPS ]])
    return np.array(x,dtype=np.float32).squeeze(),np.array(y,dtype=np.float32)

def lstm_model(x,y,is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell (HIDDEN_SIZE)
                                        for _ in range(NUM_LAYERS)])
    outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)#outputs的维度是[BATCH_SIZE,TIME_STEP,NUM_UNITS],
    # 保存了每个step中cell的输出值h。
    # output = outputs[:,-1,:]#取最后一步的值，如果要这样计算，应该是执行一次batch得到一个值，所以应该把没有整合处理的数据按时步
    # 取，然后计算出一个，再取，再算一个，

    predictions = tf.contrib.layers.fully_connected(outputs,1,activation_fn=None)

    if not is_training:
        return predictions ,None,None

    loss = tf.losses.mean_squared_error (labels= y,predictions = predictions )

    train_op = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),
                                               optimizer="Adagrad",learning_rate=0.1)
    return predictions ,loss,train_op

def train(sess,train_x,train_y):
    #这种供给数据的方式并不会改变数据的维度，
    # ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    x = np.reshape(train_x, [-1, BATCH_SIZE, TIME_STEPS])
    y = np.reshape(train_y, [-1, BATCH_SIZE, 1])
    # ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    # ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE )#当循环次数乘以batch size 大于 数据集的数量，就会报错 end of sequence ，
    # 所以，需要repeat重复使用数据，repeat().shuffle(1000)
    # x,y = ds.make_one_shot_iterator().get_next()
    print("x:", x.shape, "y:", y.shape)#xy都是三维的

    with tf.variable_scope ("model"):
        predictions,loss,train_op = lstm_model(x,y,True)#把数据一次性放进去，计算的时间特别久，循环一次，计算所有的数据，更新一次loss。

    sess.run(tf.global_variables_initializer() )
    for i in range(TRAINING_STEPS ):
        _,l = sess.run([train_op,loss])
        if i % 100 == 0:
            print("train step: " + str(i) +",loss:" + str(l))

def run_eval(sess,test_x,test_y):
    # ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    # ds = ds.batch(1 )
    # x,y = ds.make_one_shot_iterator() .get_next()
    x = np.reshape(test_x, [-1, BATCH_SIZE, TIME_STEPS])
    y = np.reshape(test_y, [-1, BATCH_SIZE, 1])
    print("x:", x.shape, "y:", y.shape)

    with tf.variable_scope("model",reuse=True):
        prediction,_,_= lstm_model(x,[0.0],False)#三维tensor(4,32,1)

    pre3 = prediction.eval(session=sess)#tensor变array
    pre2 = pre3.reshape(-1,1)  # 变为二维

    predictions = []
    labels = []

    prediction = tf.convert_to_tensor(pre2)

    # for i in range (TEST_EXAMPLES):#for 循环只是将预测结果存入到一个数组中吗，sess.run ([fetch1,fetch2])
    #只会执行和fetch相关的值，告诉网络，需要输出这个值
    predictions= sess.run(prediction)#报错的原因是[prediction,y]Y是数组而不是tensor,
        # predictions.append(p)
        # labels.append(l)
    labels = y.reshape(-1,1 )
    predictions = np.array(predictions ).squeeze()
    labels = np.array(labels).squeeze()
    print("predictions:", predictions.shape)
    print("labels:", labels.shape)
    rmse = np.sqrt(((predictions - labels )**2).mean(axis=0))
    print("Mean Square Error is: %f" % rmse)



    plt.figure()
    plt.plot(predictions ,label = 'predictions')
    plt.plot(labels,label = 'real')
    plt.legend()
    plt.show()

# test_start = delaytime (TRAINING_EXAMPLES + TIME_STEPS )* SAMPLE_GAP
# test_end = test_start + (TEST_EXAMPLES  + TIME_STEPS )* SAMPLE_GAP
#
# train_x, train_y = generate_data(np.sin(np.linspace(0,test_start,TRAINING_EXAMPLES + TIME_STEPS,
#                                                     dtype=np.float32)))
# test_x,test_y = generate_data(np.sin(np.linspace(test_start ,test_end ,TEST_EXAMPLES  + TIME_STEPS,
#                                                  dtype=np.float32)))

#先处理每一天的数据，再合为一个序列去训练神经网路
flow = maxminnorm(flow)
for i in range (26):#0-25
    a , b= generate_data(flow[i])
    # print("a:",a.shape,"b:",b.shape)
    train_x = np.vstack((train_x,a))#数组的纵向合并
    train_y = np.vstack((train_y,b))
train_x = np.delete(train_x,0,axis=0)
train_y = np.delete(train_y,0,axis=0)#第1行为0数组元素，需要将其删除
# train_x = train_x [:,np.newaxis,:]#增加维度
train_x = np.array(train_x,dtype=np.float32)
train_y = np.array(train_y,dtype=np.float32)
print("trainx:",train_x.shape,"trainy:",train_y.shape)

#测试集
for i in range (4):#0-3
    a , b= generate_data(flow[i+26])
    # print("a:",a.shape,"b:",b.shape)
    test_x = np.vstack((test_x,a))#数组的纵向合并
    test_y = np.vstack((test_y,b))
test_x = np.delete(test_x,0,axis=0)
test_y = np.delete(test_y,0,axis=0)#第1行为0数组元素，需要将其删除
# test_x = test_x [:,np.newaxis,:]
test_x = np.array(test_x,dtype=np.float32)
test_y = np.array(test_y,dtype=np.float32)
print("testx:",test_x.shape,"testy:",test_y.shape)

# print ("flowpro:",flowpro.shape,flowpro)
# train_x, train_y = generate_data(flow [0:988])
# print("train_x:",train_x.shape)
# test_x,test_y = generate_data(flow [988:1140])
# print("test_x:",test_x.shape )
# [1,2]的shape值(2,)，意思是一维数组，数组中有2个元素。
#
# [[1],[2]]的shape值是(2,1)，意思是一个二维数组，每行有1个元素。所以shape为（188,2,4）三维，每行有四个元素，一个二维是2行，一共有188个二维
#
# [[1,2]]的shape值是（1，2），意思是一个二维数组，每行有2个元素。
with tf.Session() as sess:
    train (sess,train_x,train_y)
    run_eval(sess, test_x, test_y)