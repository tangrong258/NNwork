import tensorflow as tf
import numpy as np
import csv
from tkinter import _flatten
from sklearn import preprocessing#这个函数实现对多维数组的归一化


import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib import pyplot as plt


my_matrix = np.loadtxt(open(r"C:\Users\tangrong\Desktop\flow3831.csv","rb"),delimiter=",",skiprows=0)
flow = np.array(my_matrix,dtype= np.float32)
print("flow:",flow.shape)
m = flow.shape[0]#0为行数，1为列数
flow = flow.T
flow = preprocessing.scale(flow)
# for x in flow:
#     x = float(x - np.min(flow))/(np.max(flow)- np.min(flow))
#     flow.append(x)



traindata = flow[np.arange(0,26,1)]
testdata = flow[np.arange(26,30,1)] #array和list都只能按照行来操作，需要不停转置吗
traindata = traindata.T
testdata = testdata.T
print("train:",traindata.shape,"test:",testdata.shape)




HIDDEN_SIZE = 50
NUM_LAYERS = 2

TIME_STEPS = 8
TRAINING_STEPS = 1248-TIME_STEPS
BATCH_SIZE =30#batch 是计算一次cost的集合，数据量小的时候，使用一个batch即可，

TRAINING_EXAMPLES = 1244
TEST_EXAMPLES = 192-TIME_STEPS
SAMPLE_GAP = 1


def generate_data(seq):
    x = []
    y = []
    for i in range (m-TIME_STEPS ):
        x.append([seq[i:i+TIME_STEPS]])
        y.append([seq[i+TIME_STEPS]])
    return np.array(x,dtype=np.float32).squeeze(),np.array(y,dtype=np.float32).squeeze()

def lstm_model(x,y,is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell (HIDDEN_SIZE)
                                        for _ in range(NUM_LAYERS)])
    outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    output = outputs[:,-1,:]

    predictions = tf.contrib.layers.fully_connected(output,1,activation_fn=None)
    print("predictions:",predictions.shape)

    if not is_training:
        return predictions ,None,None

    loss = tf.losses.mean_squared_error (labels= y,predictions = predictions )

    train_op = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),
                                               optimizer="Adagrad",learning_rate=0.1)
    return predictions ,loss,train_op

def train(sess,train_x,train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    ds = ds.batch(1 )#当循环次数乘以batch size 大于 数据集的数量，就会报错 end of sequence ，
    # 所以，需要repeat重复使用数据，repeat().shuffle(1000)
    x,y = ds.make_one_shot_iterator().get_next()
    print("trainx:", x.shape, "trainy:", y.shape)

    with tf.variable_scope ("model"):
        predictions,loss,train_op = lstm_model(x,y,True)

    sess.run(tf.global_variables_initializer() )
    for i in range(TRAINING_STEPS ):
        _,l = sess.run([train_op,loss])
        if i % 10 == 0:
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



train_x, train_y = generate_data(traindata)
print("train_x:",train_x.shape)
test_x,test_y = generate_data(testdata)
print("test_x:",test_x.shape )
# [1,2]的shape值(2,)，意思是一维数组，数组中有2个元素。
#
# [[1],[2]]的shape值是(2,1)，意思是一个二维数组，每行有1个元素。所以shape为（188,2,4）三维，每行有四个元素，一个二维是2行，一共有188个二维
#
# [[1,2]]的shape值是（1，2），意思是一个二维数组，每行有2个元素。
with tf.Session() as sess:
    train (sess,train_x,train_y)
    run_eval(sess,test_x,test_y)
