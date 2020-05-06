import tensorflow as tf
import pickle
import numpy as np
import os
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
time_step = 2
batch_size = 30
filters = 6
kernel_size = (2,2)
strides = (1, 1)
padding = 'same'
data_format = 'channels_last'
input_shape = (batch_size,time_step,featime[0].shape[0],featime[0].shape[1],featime[0].shape[2])
dilation_rate = (1, 1)
activation = 'a(x) = x'
recurrent_activation = 'a(x) = x'
use_bias = True
kernel_initializer = 'glorot_uniform'
recurrent_initializer = 'orthogonal'
bias_initializer = 'zeros'
unit_forget_bias = True


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

input_size = n
hidden_size = 200
layer_num = 1
output_size = ((1,7))
LR =0.1
train_end = int(0.8 * m)
test_end = int(m)
train_step = 2000
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
                 time_step,
                 filters,
                 kernel_size ,
                 strides,
                 padding,
                 data_format,
                 dilation_rate ,
                 # activation ,
                 # recurrent_activation ,
                 # use_bias ,
                 # kernel_initializer ,
                 # recurrent_initializer ,
                 # bias_initializer ,
                 # unit_forget_bias ,
                 input_size, output_size, hidden_size, layer_num):
        self.batch_size = batch_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding =padding
        self.data_format =data_format
        self.dilation_rate =dilation_rate
        # self.activation =activation
        # self.recurrent_activation = recurrent_activation
        # self.use_bias =use_bias
        # self.kernel_initializer =kernel_initializer
        # self.recurrent_initializer = recurrent_initializer
        # self.bias_initializer =bias_initializer
        # self.unit_forget_bias = unit_forget_bias
        self.time_step = time_step
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num

        with tf.name_scope('inputs'):
            self.xs1 = tf.placeholder(tf.float32, input_shape, name='xs1')
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
        cell = tf.keras.layers.ConvLSTM2D(self.filters ,
                                self.kernel_size,
                                self.strides,
                                self.padding,
                                self.data_format,
                                self.dilation_rate)#a[:-1]除了最后一个取全部
                                          # self.activation,
                                          # self.recurrent_activation,
                                          # self.use_bias,
                                          # self.kernel_initializer,
                                          # self.recurrent_initializer,
                                          # self.bias_initializer,
                                          # self.unit_forget_bias,
        self.init_state1 = None
        # 按照时间序列展开
        self.outputs1, self.final_state1 = tf.nn.dynamic_rnn(cell, self.input_x1, initial_state=self.init_state1, time_major=False)
        #这里就很特殊，他不要求input去变为二维或者怎么样，input的shape为[batchsize,timestep,width,height,inputchannels],然后每一个timestep对应一个cell，所以一个timestep
        #的输入就是[batchsize,width,height,inputchannels],这刚好就是cell的输入，然后一个cell就能求一个state[batchsize,width,height,outputchannels]
        # 和总的序列的output[batchsize,timestep,width,height,outputchannels]
        self.output1 = self.outputs1[:, -1, :,:,:]#[batch_size,width,height,output_channels]



    def input_layer2(self,):
        self.input_x2 = tf.reshape(self.xs2, [-1, self.input_size])  #因为要给input分配权重，所以将input_size
        Ws_in = self._weight_variable([self.input_size, self.hidden_size])
        bs_in = self._bias_variable([self.hidden_size])
        with tf.name_scope('Wx_plus_b'):
            self.output_x2 = tf.matmul(self.input_x2, Ws_in) + bs_in
        self.output_x2 = tf.reshape(self.output_x2, [-1, self.time_step, self.hidden_size])#转换成cell要求的格式

    def LSTMcell(self):
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                                            for _ in range(self.layer_num)])#定义一个单元，就是一个timestep
        self.init_state2 = cell.zero_state(self.batch_size, dtype=tf.float32)
        self.outputs2, self.final_state2 = tf.nn.dynamic_rnn(cell, self.output_x2, initial_state=self.init_state2, time_major=False)#
        self.output2 = self.outputs2[:, -1, :]#[batch_size,hidden_size]

    def feature_layer(self):#有多重全连接的方法
        self.output_x1 = tf.reshape(self.output1, [self.batch_size, -1])
        self.output_x2 = self.output2
        with tf.name_scope('fea1'):
            self.Ws_out1 = self._weight_variable([self.output_x1.shape[1],32],name = 'ws1')
            self.bs_out1 = self._bias_variable([32],name ='bs1')
        with tf.name_scope('fea2'):
            self.Ws_out2 = self._weight_variable([self.hidden_size, 32],name = 'ws2')
            self.bs_out2 = self._bias_variable([32],name = 'bs2')
        with tf.name_scope('Wx_plus_b'):
            self.fealay1 = tf.matmul(self.output_x1, self.Ws_out1) + self.bs_out1
            self.fealay2 = tf.matmul(self.output_x2, self.Ws_out2) + self.bs_out2
            self.fea_out = tf.add(self.fealay1,self.fealay2)
    def fulcon_layer(self):
        self.Ws_out = self._weight_variable([32, self.output_size[0] * self.output_size[1]])
        self.bs_out = self._bias_variable([self.output_size[0] * self.output_size [1]])
        self.pred = tf.matmul(self.fea_out, self.Ws_out) + self.bs_out#[batchsize,width*height]
        self.pred = tf.nn.dropout(self.pred, self.keep_prob)
        return self.pred
    def compute_cost(self):
        # self.loss = tf.losses.mean_squared_error(predictions = self.pred,labels = self.ys)
        mse = [[]for i in range(7)]#np.zeros(self.batch_size,dtype = 'float32')
        # for i in range(0, self.batch_size):
        #     for j in range(0, self.ys.shape[1]):
        #         mse[i, j] = np.square(self.ys[i, j] - self.pred[i, j])
        # self.loss = np.mean(mse)
        for i in range(0, self.ys.shape[1]):
            a = tf.reduce_mean(tf.square(self.ys[:,i] - self.pred[:,i]))
            mse[i].append(a)
        self.loss = tf.reduce_mean(mse)
        # self.train_op = tf.contrib.layers.optimize_loss(self.loss, tf.train.get_global_step(),optimizer="Adam", learning_rate = LR)
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
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
        xs = np.arange(0,7)
        bar_width = 0.3
        print("teststep:",i)
        for j in range(batch_size):
            plt.bar(xs, y[j], bar_width,label = 'real',align="center",color = "g")
            plt.bar(xs + bar_width, pred[j], bar_width,label = 'pred', align="center",color = "r")
            plt.legend()
            plt.savefig(r'D:\ZSNJAP01\flight\prediction\linedlyt' + '\\' + 'bar'+str(i*batch_size)+str(j) + '.png')
            plt.close()

if __name__ == '__main__':
    model = convLSTM2(batch_size,
                 time_step,
                 filters,
                 kernel_size ,
                 strides,
                 padding,
                 data_format,
                 dilation_rate ,
                 input_size, output_size, hidden_size, layer_num)
    # activation ,
    # recurrent_activation ,
    # use_bias ,
    # kernel_initializer ,
    # recurrent_initializer ,
    # bias_initializer ,
    # unit_forget_bias ,
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter("logs", sess.graph)
    train_x1, train_y = generate_data1(featime[0:train_end])
    train_x2 = generate_data2(input2[0:train_end])
    test_x1, test_y = generate_data1(featime[train_end:test_end])
    test_x2 = generate_data2(input2[train_end:test_end])
    train(sess,train_x1,train_y,train_x2)
    test(sess,test_x1,test_y,test_x2)