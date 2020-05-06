import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import  os
from sklearn.preprocessing import MinMaxScaler

b = os.path.exists(r"D:\ZSNJAP01\flight\prediction\linedlytFeLSTM")
if b :
    print("path exist")
else:
    os.makedirs(r'D:\ZSNJAP01\flight\prediction\linedlytFeLSTM')

# with open(r"D:\ZSNJAP01\flight\feature\featuretime.pkl",'rb') as f:
#     featime = pickle.load(f)#list dim =[3000,2,1,64]
with open(r"D:\ZSNJAP01\flight\delaydata\lineavgt.pkl",'rb') as f:
    featime = pickle.load(f) #ndarray dim = [3623,4,5]
featime2d = np.reshape(featime,[-1,len(featime[0])*len(featime[0][0])])#dim =[3000,128],*len(featime[0][0][0])
scaler1= MinMaxScaler(feature_range=(0, 1))
featime = scaler1.fit_transform(featime2d).squeeze()

with open(r"D:\ZSNJAP01\flight\weather\weather5.pkl",'rb') as f:
    wea = pickle.load(f)#dim =[3624]

with open(r"D:\ZSNJAP01\flight\delaydata\spot.pkl",'rb') as f:
    spot = pickle.load(f)#dim = [n,9]

with open(r"D:\ZSNJAP01\flight\delaydata\linedlyt.pkl",'rb') as f:
    linedlyt = pickle.load(f)#[3623,4,5]
column = linedlyt.shape[1]*linedlyt.shape[2]
linedlyt = np.reshape(linedlyt,[-1,column])

minrow = min(len(featime),len(wea),len(spot),len(linedlyt))
m = minrow
n = len(featime[0]) + 1 + len(spot[0])
input = np.zeros((m, n))
output = np.zeros((m,column))
for i in range(m):
    for j in range(0,len(featime[0])):
        input[i,j] = featime[i][j]
    for j in range(len(featime),n-1):
        input[i, j] = spot[i,j]
    for j in range(n-1, n ):
        input[i, j] = wea[i]
    for j in range(column):
        output[i, j] = linedlyt[i,j]

time_step = 4
input_size = n
output_size =5*4#预测像素点（航段延误时间）
hidden_size = 300
layer_num = 2
batch_size = 30
LR =0.1
train_end = int(0.8 * m)
test_end = int(m)
train_step = 2000
test_step = (test_end - train_end - time_step) // batch_size


def generate_data(seq1,seq2):#seq是一个二维数组
    x = []
    y = []
    for i in range(len(seq1) - time_step):
        x.append([seq1[i:i + time_step]])#x必须是二维的，因为是len(seq) - time_step个序列，所以seq[i:i + time_step]要变为list加入x
        y.append([seq2[i + time_step]])#预测值是图片的像素，为5*4=20
    return np.array(x, dtype=np.float32).squeeze(), np.array(y, dtype=np.float32).squeeze()

class LSTMRNN(object):
    def __init__(self,batch_size, time_step, input_size, output_size, hidden_size, layer_num):
        self.time_step = time_step
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None , time_step, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None , output_size], name='ys')
            self.keep_prob = tf.placeholder(tf.float32)
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()


    def add_input_layer(self,):
        input_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  #因为要给input分配权重，所以将input_size
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.hidden_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.hidden_size])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            input_y = tf.matmul(input_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.input_y = tf.reshape(input_y, [-1, self.time_step, self.hidden_size], name='2_3D')#转换成cell要求的格式

    def add_cell(self):
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                                            for _ in range(self.layer_num)])#定义一个单元，就是一个timestep
        self.init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        self.outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.input_y, initial_state=self.init_state, time_major=False)#
        #按照时间序列展开
        self.output = self.outputs[:, -1, :]

    def add_output_layer(self):
        # shape = (batch * steps, hidden_size)
        output_x = tf.reshape(self.output, [-1, self.hidden_size], name='2_2D')
        Ws_out = self._weight_variable([self.hidden_size, self.output_size])
        bs_out = self._bias_variable([self.output_size])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.nn.sigmoid(tf.matmul(output_x, Ws_out) + bs_out)
            self.pred = tf.nn.dropout(self.pred,self.keep_prob)
            #[batch_size*time_step.output_size]表示每一步都输出，
            #[batch_size.output_size]表示只输出最后一步
        return self.pred

    def compute_cost(self):
        self.loss = tf.losses.mean_squared_error(labels = self.ys, predictions = self.pred)
        # self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.pred)#交叉熵损失函数,适用于分类
        # self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)
        #有些操作没有返回值，此时该对象就是一个Operation对象，就相当于它在优化使得loss最小，这就是他表示的operation，
        # 但是他不返回这个loss，
        self.train_op = tf.contrib.layers.optimize_loss(self.loss, tf.train.get_global_step(),optimizer="Adam", learning_rate = LR)
        #返回值是结果，是一个tensor，他是operation返回值中的一个，他返回的就是loss，返回就会形成一个位置存放loss，
        # 所以，他不需要tf.summary.scalar('loss', loss)去再定义一个scalar存放loss
        return  self.loss, self.train_op

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

def train(sess,train_x,train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    x, y = ds.make_one_shot_iterator().get_next()
    x = x.eval(session=sess)  # [30,4,138]
    y = y.eval(session=sess)  # [30,20]
    batch_start = time_step
    for i in range(train_step):
        if i == 0:
            feed_dict = {
                    model.xs: x,
                    model.ys: y,
                    model.keep_prob:0.8
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: x,
                model.ys: y,
                model.keep_prob: 0.8,
                model.init_state: state }

        _, loss,state, pred = sess.run([model.train_op, model.loss, model.final_state, model.pred],feed_dict=feed_dict)

        xs = np.arange(batch_start, batch_start + batch_size * 1)
        batch_start = batch_start + batch_size
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()#听说这个因为TensorFlow更新太频繁的原因，总是报错nonetype
        if i % 100 ==0:
            # plt.plot(xs, y.T[0].flatten(), 'r', xs, pred.T[0].flatten(), 'b--')
            # plt.draw()
            # plt.pause(0.01)
            print('lost: '+ str(i), loss)
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)


def test(sess,test_x,test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(batch_size)
    x, y = ds.make_one_shot_iterator().get_next()
    x = x.eval(session=sess)
    y = y.eval(session=sess)
    for i in range(test_step):
        if i == 0:
            feed_dict = {
                    model.xs: x,
                    model.keep_prob:1.0
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: x,
                model.keep_prob: 1.0,
                model.init_state: state}

        state, pred = sess.run([model.final_state, model.pred],feed_dict=feed_dict)
        xs = np.arange(0,20)
        bar_width = 0.3
        print("teststep:",i)
        for j in range(batch_size):
            plt.bar(xs, y[j], bar_width,label = 'real',align="center")
            plt.bar(xs + bar_width, pred[j], bar_width,label = 'pred', align="center")
            plt.savefig(r'D:\ZSNJAP01\flight\prediction\linedlytFeLSTM' + '\\' + 'bar'+str(i*batch_size + j)+ '.png')
            plt.close()

if __name__ == '__main__':
    model = LSTMRNN(batch_size,time_step, input_size, output_size, hidden_size, layer_num)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter("logs", sess.graph)
    train_x, train_y = generate_data(input[0:train_end],output[0:train_end])
    test_x, test_y = generate_data(input[train_end:test_end], output[train_end:test_end])
    train(sess,train_x,train_y)
    test(sess,test_x,test_y)


