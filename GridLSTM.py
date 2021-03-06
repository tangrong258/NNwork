import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import os
import tensorflow.contrib as contrib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


howlong = "30min"
sp = 30
rootfrom = r"D:\ZSNJAP01\flight\delaydata" + "\\"+ howlong

with open(rootfrom + "\\" + "lineavgt.pkl",'rb') as f:
    featime = pickle.load(f)

featime2d = np.reshape(featime,[-1,featime.shape[1] * featime.shape[2]])

for i in range (featime2d.shape[1]):
    for j in range(featime2d.shape[0]):
        if featime2d[j,i] >= 10.0:
            featime2d[j,i] = np.mean(featime2d[:,i])
        if featime2d[j,i] <= 0.0:
            featime2d[j, i] = 0.01

scaler1= MinMaxScaler(feature_range=(0, 1))
featime2d = scaler1.fit_transform(featime2d).squeeze()

featime = np.reshape(featime2d,[-1,featime.shape[1],featime.shape[2]])
featime = featime[:,:,:,np.newaxis]

num_units = 100
time_step = 6
batch_size = int(24*(60/sp))
input1_shape = featime[0].shape
name = "grid_lstm_cell"
num_frequency_blocks = [1,1]



with open(rootfrom + "\\" + "weather5.pkl",'rb') as f:
    wea = pickle.load(f)
with open(rootfrom + "\\" + "spot.pkl",'rb') as f:
    spot = pickle.load(f)
minrow = min(len(featime),len(wea),len(spot))
m = minrow
n = 1 + len(spot[0])
input2 = np.zeros((m, n))

for i in range(m):
    for j in range(0,1):
        input2[i, j] = wea[i]
    for j in range(1,n):
        input2[i, j] = spot[i][j-1]
scaler2 = MinMaxScaler(feature_range=(0, 1))
input2 = scaler2.fit_transform(input2).squeeze()

input2_size = n
hidden_size = 50
layer_num = 2
output_size = ((1,1))
train_end = int((31+28+31)*24*(60/sp)) - time_step
test_end = int(m) - time_step
train_step = 4000
test_step = (test_end - train_end) // batch_size


def generate_data1(seq):#seq[n,4,5,1]
    x = []
    y = []
    for i in range(len(seq) - time_step):
        x.append([seq[i:i + time_step]])
        y.append([seq[i + time_step]])
    xnp = np.array(x, dtype = np.float32)#[batchsize,1,timestep,w,h,inchan]
    ynp1 = np.array(y, dtype = np.float32)#[batchsize,1,w,h,inc]
    ynp1 = np.reshape(ynp1, [-1, ynp1.shape[2] * ynp1.shape[3]])
    ynp = np.zeros((ynp1.shape[0],7),dtype = 'float32')
    ynp[:, 0] = ynp1[:, 2]
    ynp[:, 1] = ynp1[:, 7]
    ynp[:, 2] = ynp1[:, 12]
    ynp[:, 3] = ynp1[:, 15]
    ynp[:, 4] = ynp1[:, 16]
    ynp[:, 5] = ynp1[:, 18]
    ynp[:, 6] = ynp1[:, 19]
    return np.reshape(xnp,[xnp.shape[0],-1,xnp.shape[3],xnp.shape[4],xnp.shape[5]]),ynp

def generate_data2(seq):#seq是一个二维数组
    x = []
    for i in range(len(seq) - time_step):
        x.append([seq[i:i + time_step]])
    return np.array(x, dtype=np.float32).squeeze()


class GridLSTM(object):
    def __init__(self,num_units, num_frequency_blocks,
                 batch_size,
                 input1_shape,
                 name,
                 time_step, input2_size, output_size, hidden_size, layer_num,opt):
        self.num_units = num_units
        self.num_frequency_blocks = num_frequency_blocks
        self.batch_size = batch_size
        self.input1_shape = input1_shape
        self.name = name
        self.time_step = time_step
        self.input2_size = input2_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.opt = opt
        with tf.name_scope('inputs'):
            self.xs1 = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.input1_shape[0], self.input1_shape[1], 1], name='xs1')
            self.ys = tf.placeholder(tf.float32, [self.batch_size , self.output_size[0] * self.output_size[1]], name='ys')
            self.xs2 = tf.placeholder(tf.float32, [self.batch_size , self.time_step, self.input2_size], name='xs2')
            self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('in_hidden1'):
            self.input_layer1()
        with tf.variable_scope('in_hidden2'):
            self.input_layer2()
        with tf.variable_scope('GridLSTM_cell'):
            self.GridLSTMCell()
        with tf.variable_scope('LSTM_cell'):
            self.LSTMcell()
        with tf.variable_scope('out_hidden'):
            self.feature_layer()
        with tf.variable_scope('fully_connected'):
            self.fulcon_layer()
        with tf.name_scope('cost'):
            self.compute_cost()



    def input_layer1(self):
        self.input_x1 = self.xs1


    def GridLSTMCell(self):
        cell = contrib.rnn.GridLSTMCell(num_units = self.num_units, num_frequency_blocks = self.num_frequency_blocks)
        self.init_state1 = cell.zero_state(self.batch_size, dtype=tf.float32)

        self.outputs1, self.final_state1 = tf.nn.dynamic_rnn(cell, self.input_x1, initial_state=self.init_state1, time_major=False)

        self.output1 = tf.nn.relu(self.outputs1[:, -1, :, :, :])
    def input_layer2(self):

        self.output_x2 = self.xs2

    def LSTMcell(self):
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                                            for _ in range(self.layer_num)])#定义一个单元，就是一个timestep
        self.init_state2 = cell.zero_state(self.batch_size, dtype=tf.float32)
        self.outputs2, self.final_state2 = tf.nn.dynamic_rnn(cell, self.output_x2, initial_state=self.init_state2, time_major=False)#
        self.output2 = self.outputs2[:, -1, :]#[batch_size,hidden_size]

    def feature_layer(self):#有多重全连接的方法
        self.output_x1 = tf.reshape(self.output1, [self.batch_size, -1])
        self.output_x2 = self.output2
        self.fea_out = tf.concat([self.output_x1, self.output_x2],1)

    def fulcon_layer(self):
        self.pred = tf.contrib.layers.fully_connected(self.fea_out, self.output_size[0] * self.output_size[1], activation_fn=None)
        self.pred = tf.nn.leaky_relu(self.pred, alpha=0.2, name=None)
        self.pred = tf.nn.dropout(self.pred, self.keep_prob)
        return self.pred

    def compute_cost(self):

        mse = [[] for i in range(output_size[0] * output_size[1])]
        for i in range(0, self.ys.shape[1]):
            a = tf.reduce_mean(tf.square(self.ys[:, i] - self.pred[:, i]))
            mse[i].append(a)
        self.loss = tf.reduce_mean(mse)
        global_step = tf.Variable(0)
        LR = tf.train.exponential_decay(0.1, global_step, int(m / batch_size), 0.96, staircase=True)
        self.train_op = tf.contrib.layers.optimize_loss(self.loss, tf.train.get_global_step(), optimizer = self.opt, learning_rate=LR)
        return self.loss, self.train_op

def train(sess,x1,y,x2):
    ds = tf.data.Dataset.from_tensor_slices((x1,y,x2))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    x1,y,x2 = ds.make_one_shot_iterator().get_next()
    x1 = x1.eval(session = sess)
    y = y.eval(session = sess)
    x2 = x2.eval(session = sess)
    with tf.variable_scope("mulmodel", reuse=tf.AUTO_REUSE):
        sta1, sta2, lossor, train_op = model.final_state1, model.final_state2,model.loss, model.train_op
    for i in range(train_step):
        if i == 0:
            feed_dict = {
                model.xs1: x1,
                model.xs2:x2,
                model.ys:y,
                model.keep_prob: 0.8
            }
        else:
            feed_dict = {
                model.xs1: x1,
                model.xs2:x2,
                model.ys:y,
                model.keep_prob: 0.8,
                model.init_state1: state1,
                model.init_state2: state2
            }
        state1, state2, loss, _ = sess.run([sta1, sta2, lossor, train_op], feed_dict=feed_dict)
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        if i % 100 ==0:
            print('lost: ' + str(i), loss)
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)


def test(sess,x1,y,x2):
    ds = tf.data.Dataset.from_tensor_slices((x1, y, x2))
    ds = ds.batch(batch_size)
    x1, y, x2 = ds.make_one_shot_iterator().get_next()
    x1 = x1.eval(session=sess)
    y = y.eval(session=sess)
    x2 = x2.eval(session=sess)

    with tf.variable_scope("mulmodel", reuse=tf.AUTO_REUSE):
        sta1, sta2, predor = model.final_state1, model.final_state2, model.pred
    for i in range(test_step):
        if i == 0:
            feed_dict = {
                model.xs1: x1,
                model.xs2: x2,
                model.ys: y,
                model.keep_prob: 1.0
            }
        else:
            feed_dict = {
                model.xs1: x1,
                model.xs2: x2,
                model.ys: y,
                model.keep_prob: 1.0,
                model.init_state1: state1,
                model.init_state2: state2
            }

        state1, state2, pred = sess.run([sta1, sta2, predor], feed_dict=feed_dict)
        ys = y
        mse = np.zeros((ys.shape[0], ys.shape[1]))
        mae = np.zeros((ys.shape[0], ys.shape[1]))
        mape = np.zeros((ys.shape[0], ys.shape[1]))
        for k in range(0, ys.shape[1]):
            for l in range(0, ys.shape[0]):
                a = np.square(ys[l, k] - pred[l, k])
                c = np.abs(ys[l, k] - pred[l, k])
                b = np.abs(ys[l, k] - pred[l, k]) / (ys[l, k])
                mse[l, k] = a
                mae[l, k] = c
                mape[l, k] = b
        MSE = np.mean(mse, axis=1)
        MAE = np.mean(mae, axis=1)
        MAPE = np.mean(mape, axis=1)
        ERROR = [MSE, MAE, MAPE]
        print("teststep:", i, ERROR)
        for e in range(0, 3):
            errorcsv[i * batch_size:(i + 1) * batch_size, e + 3 * p] = ERROR[e]


if __name__ == '__main__':

    x1, y7 = generate_data1(featime)
    train_x1 = x1[0:train_end]
    train_y7 = y7[0:train_end]
    x2 = generate_data2(input2)
    train_x2 = x2[0:train_end]
    train_y = np.zeros((train_y7.shape[0], 1), dtype=np.float32)
    for i in range(train_y7.shape[0]):
        train_y[i,0] = train_y7[i,0]

    test_x1 = x1[train_end:test_end]
    test_y7 = y7[train_end:test_end]
    test_x2 = x2[train_end:test_end]
    test_y = np.zeros((test_y7.shape[0],1),dtype = np.float32)
    for i in range(test_y7.shape[0]):
        test_y[i,0] = test_y7[i, 0]

    root = r"D:\ZSNJAP01\flight\prediction\convLSTM" + "\\" + howlong
    sub = "lineavgt"
    optimizer = ["SGD", "Adagrad", "Momentum"]
    errorcsv = np.zeros((test_step * batch_size, len(optimizer) * 3))
    for p in range(0, len(optimizer)):
        opt = optimizer[p]
        b = os.path.exists(root + "\\" + sub + "\\" + opt)
        if b:
            print("path exist")
        else:
            os.makedirs(root + "\\" + sub + "\\" + opt)

        tf.reset_default_graph()
        model = GridLSTM(num_units, num_frequency_blocks, batch_size, input1_shape, name,
                          time_step, input2_size, output_size, hidden_size, layer_num, opt)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter("logs", sess.graph)
        train(sess, train_x1, train_y,train_x2)
        test(sess, test_x1, test_y, test_x2)


    #误差统计
    sixto24error = np.zeros((36*test_step, 9))
    for i in range(0,test_step):
        sixto24error[i*36:(i+1)*36] = errorcsv[48*i+12:48*(i+1)]

    dt =  pd.DataFrame(sixto24error, columns = [optimizer[0],'','',optimizer[1],'','',optimizer[2],'',''])
    dt.to_csv(root + "\\" + sub + "\\" +"error2.csv")