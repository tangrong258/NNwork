import tensorflow as tf
import pickle
import numpy as np
import os
import tensorflow.contrib as contrib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

b = os.path.exists(r"D:\ZSNJAP01\flight\prediction\linedlyt")
if b :
    print("path exist")
else:
    os.makedirs(r'D:\ZSNJAP01\flight\prediction\linedlyt')

with open(r"D:\ZSNJAP01\flight\delaydata\30min\lineavgt.pkl",'rb') as f:
    featime = pickle.load(f) #ndarray dim = [3623,4,5]
featime2d = np.reshape(featime,[-1,featime.shape[1] * featime.shape[2]])
scaler1= MinMaxScaler(feature_range=(0, 1))
featime2d = scaler1.fit_transform(featime2d).squeeze()
featime = np.reshape(featime2d,[-1,featime.shape[1],featime.shape[2]])
featime = featime[:,:,:,np.newaxis]
time_step = 6
batch_size = 20
conv_ndims = 2
input_shape = featime[0].shape#[4,5,1]
kernel_shape = (2,2)
output_channels = 2
use_bias=True,
skip_connection=False,
forget_bias=1.0,
initializers=None,
name="conv_lstm_cell"




with open(r"D:\ZSNJAP01\flight\delaydata\30min\weather5.pkl",'rb') as f:
    wea = pickle.load(f)#dim =[3624]
with open(r"D:\ZSNJAP01\flight\delaydata\30min\spot.pkl",'rb') as f:
    spot = pickle.load(f)#dim = [3623,9]
minrow = min(len(featime),len(wea),len(spot))
m = minrow
n = 1 + len(spot[0])
input2 = np.zeros((m, n))
for i in range(m):
    for j in range(0,1):
        input2[i, j] = wea[i]
    for j in range(1,n):
        input2[i, j] = spot[i][j-1]
scaler2= MinMaxScaler(feature_range=(0, 1))
input2 = scaler2.fit_transform(input2).squeeze()


time_step = 3
input_size = n
hidden_size = 200
layer_num = 1
batch_size = 30
output_size = ((1,7))
train_end = int(0.8 * m)
test_end = int(m)
train_step = 1000
test_step = (test_end - train_end - time_step) // batch_size


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


class convLSTM2(object):
    def __init__(self,
                 batch_size,
                 conv_ndims,
                 input_shape,
                 output_channels,
                 kernel_shape,
                 use_bias,
                 skip_connection,
                 forget_bias,
                 initializers,
                 name,
                 time_step, input_size, output_size, hidden_size, layer_num):

        self.batch_size = batch_size
        self.conv_ndims = conv_ndims
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.use_bias = use_bias
        self.skip_connection = skip_connection
        self.forget_bias = forget_bias
        self.initializers = initializers
        self.name = name
        self.time_step = time_step
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num

        with tf.name_scope('inputs'):
            self.xs1 = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.input_shape[0],self.input_shape[1],1], name='xs1')
            self.ys = tf.placeholder(tf.float32, [self.batch_size , self.output_size[0] * self.output_size[1]], name='ys')
            self.xs2 = tf.placeholder(tf.float32, [self.batch_size , self.time_step, self.input_size], name='xs2')
            self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('in_hidden1'):
            self.input_layer1()
        with tf.variable_scope('in_hidden2'):
            self.input_layer2()
        with tf.variable_scope('convLSTM_cell'):
            self.convLSTMcell()
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

    def convLSTMcell(self):
        cell = contrib.rnn.ConvLSTMCell(self.conv_ndims, self.input_shape, self.output_channels, self.kernel_shape)#a[:-1]除了最后一个取全部
        self.init_state1 = cell.zero_state(self.batch_size, dtype=tf.float32)
        # 按照时间序列展开
        self.outputs1, self.final_state1 = tf.nn.dynamic_rnn(cell, self.input_x1, initial_state=self.init_state1, time_major=False)

        #这里就很特殊，他不要求input去变为二维或者怎么样，input的shape为[batchsize,timestep,width,height,inputchannels],然后每一个timestep对应一个cell，所以一个timestep
        #的输入就是[batchsize,width,height,inputchannels],这刚好就是cell的输入，然后一个cell就能求一个state[batchsize,width,height,outputchannels]
        # 和总的序列的output[batchsize,timestep,width,height,outputchannels]
        self.output1 = tf.nn.relu(self.outputs1[:, -1, :,:,:])#[batch_size,width,height,output_channels]



    def input_layer2(self,):
        self.input_x2 = tf.reshape(self.xs2, [-1, self.input_size])  #因为要给input分配权重，所以将input_size
        Ws_in = self._weight_variable([self.input_size, self.hidden_size])
        bs_in = self._bias_variable([self.hidden_size])
        with tf.name_scope('Wx_plus_b'):
            self.output_x2 = tf.matmul(self.input_x2, Ws_in) + bs_in
        self.output_x2 = tf.reshape(self.output_x2, [-1, self.time_step, self.hidden_size])#转换成cell要求的格式

    def LSTMcell(self):
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,activation = 'tanh')
                                            for _ in range(self.layer_num)])#定义一个单元，就是一个timestep
        self.init_state2 = cell.zero_state(self.batch_size, dtype=tf.float32)
        self.outputs2, self.final_state2 = tf.nn.dynamic_rnn(cell, self.output_x2, initial_state=self.init_state2, time_major=False)#
        self.output2 = tf.nn.relu(self.outputs2[:, -1, :])#[batch_size,hidden_size]


    def feature_layer(self):#有多重全连接的方法
        self.output_x1 = tf.reshape(self.output1, [self.batch_size, -1])
        self.output_x2 = self.output2
        with tf.name_scope('fea1'):
            self.Ws_out1 = self._weight_variable([self.output_x1.shape[1],20],name = 'ws1')
            # self.bs_out1 = self._bias_variable([20],name ='bs1')
        with tf.name_scope('fea2'):
            self.Ws_out2 = self._weight_variable([self.hidden_size, 20], name='ws2')
            # self.bs_out2 = self._bias_variable([20],name = 'bs2')
        with tf.name_scope('fea_out'):
            # self.fealay1 = tf.matmul(self.output_x1, self.Ws_out1) + self.bs_out1
            # self.fealay2 = tf.matmul(self.output_x2, self.Ws_out2) + self.bs_out2
            self.bs_out = self._bias_variable([20], name='bs_out')
            self.fea_out = tf.nn.relu(tf.add(tf.matmul(self.output_x1, self.Ws_out1) ,tf.matmul(self.output_x2, self.Ws_out2))+ self.bs_out)


    def fulcon_layer(self):
        self.Ws_out = self._weight_variable([20, self.output_size[0] * self.output_size[1]])
        self.bs_out = self._bias_variable([self.output_size[0] * self.output_size [1]])
        self.pred = tf.nn.elu(tf.matmul(self.fea_out, self.Ws_out) + self.bs_out)#[batchsize,width*height]
        self.pred = tf.nn.dropout(self.pred, self.keep_prob)
        return self.pred

    def compute_cost(self):
        mse = [[] for i in range(7)]
        for i in range(0, self.ys.shape[1]):
            a = tf.reduce_mean(tf.square(self.ys[:,i] - self.pred[:,i]))
            mse[i].append(a)
        self.loss = tf.reduce_mean(mse)
        # self.loss = tf.reduce_mean(tf.square(self.pred - self.ys))
        global_step = tf.Variable(0)
        LR = tf.train.exponential_decay(0.1,global_step,int(m / batch_size),0.96,staircase= True)
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss,global_step= global_step)
        return  self.loss, self.train_op

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable( shape=shape,initializer=initializer,name=name)


def train(sess,x1,y,x2):
    ds = tf.data.Dataset.from_tensor_slices((x1,y,x2))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    x1,y,x2 = ds.make_one_shot_iterator().get_next()
    x1 = x1.eval(session = sess)
    y = y.eval(session = sess)
    x2 = x2.eval(session = sess)
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

        state1,state2,pred,loss,_ = sess.run([model.final_state1,model.final_state2, model.pred,model.loss,model.train_op],feed_dict=feed_dict)
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        if i % 100 ==0:
            print('lost: '+ str(i), loss)
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)


def test(sess,x1,y,x2):
    ds = tf.data.Dataset.from_tensor_slices((x1, y, x2))
    ds = ds.batch(batch_size)
    x1, y, x2 = ds.make_one_shot_iterator().get_next()
    x1 = x1.eval(session=sess)
    y = y.eval(session=sess)
    x2 = x2.eval(session=sess)
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
        state1, state2, pred = sess.run([model.final_state1, model.final_state2, model.pred], feed_dict=feed_dict)
        xs = np.arange(0,output_size[0] * output_size[1])
        bar_width = 0.3
        print("teststep:",i)
        for j in range(batch_size):
            plt.bar(xs, y[j], bar_width,label = 'real',align="center",color = "g")
            plt.bar(xs + bar_width, pred[j], bar_width,label = 'pred', align="center",color = "r")
            plt.legend()
            plt.savefig(r'D:\ZSNJAP01\flight\prediction\linedlyt' + '\\' + 'bar'+str(i*batch_size+j)+ '.png')
            plt.close()

if __name__ == '__main__':
    model = convLSTM2(batch_size,conv_ndims,input_shape,output_channels,kernel_shape,use_bias,skip_connection,forget_bias,initializers,name,
                       time_step, input_size, output_size, hidden_size, layer_num)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter("logs", sess.graph)
    train_x1, train_y = generate_data1(featime[0:train_end])
    # train_y = train_yz[:,0,np.newaxis]
    train_x2 = generate_data2(input2[0:train_end])
    test_x1, test_y = generate_data1(featime[train_end:test_end])
    # test_y = test_yz[:,0,np.newaxis]
    test_x2 = generate_data2(input2[train_end:test_end])
    train(sess,train_x1,train_y,train_x2)
    test(sess,test_x1,test_y,test_x2)

