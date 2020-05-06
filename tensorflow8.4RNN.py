import tensorflow as tf
import numpy as np
import csv
from tkinter import _flatten

import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib import pyplot as plt

csvFile = open(r'C:\Users\tangrong\Desktop\ZSNJ201805.csv', "r")
reader = csv.reader(csvFile)#csv.reader()返回是一个迭代类型，需要通过循环或迭代器访问
list1=[]
list2=[]
#全部读进来，再操作
print(reader)
for row in reader:
    #list1 += [row[1],row[2],row[3]]
    list2 += [row[1]]

#list1 = np.array(list1).reshape(len(list1)//3,3)
#list2 = np.array(list2).reshape(len(list2),1)
delaytime = list(_flatten(list2))
delaytime = np.array(delaytime,dtype= np.float32)
flow = []
for x in delaytime:
    x = float(x - np.min(delaytime))/(np.max(delaytime )- np.min(delaytime))
    flow.append(x)
print("delaytime:", flow)


HIDDEN_SIZE = 50
NUM_LAYERS = 2

TIME_STEPS = 4
TRAINING_STEPS = 1244
BATCH_SIZE = 48

TRAINING_EXAMPLES = 1244
TEST_EXAMPLES = 188

lr = 0.1
batch_size = 1
n_inputs = 1
n_steps = 4
n_hidden_units =50
n_classes = 1


def generate_data(seq):
    x = []
    y = []
    for i in range (len(seq)-TIME_STEPS ):
        x.append([seq[i:i+TIME_STEPS ]])#append函数是在末尾添加，所以x现在还是
        y.append([seq[i+TIME_STEPS ]])
    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)#这里会产生3维的x([行（输入值的种类，一般就1吧）*列timesteps]*len-timesteps)
#
# def lstm_model(x,y,is_training):
#     cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell (HIDDEN_SIZE)
#                                         for _ in range(NUM_LAYERS)])
#     outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
#     output = outputs[:,-1,:]
#
#     predictions = tf.contrib.layers.fully_connected(output,1,activation_fn=tf.nn.leaky_relu)
#
#     if not is_training:
#         return predictions ,None,None
#
#     loss = tf.losses.mean_squared_error (labels= y,predictions = predictions )
#
#     train_op = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),
#                                                optimizer="Adagrad",learning_rate=0.1)
#     return predictions ,loss,train_op

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))}

def RNN(X, y,weights, biases,is_training):
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])#-1就是把128*28总起来
    #
    # # into hidden
    X_in = tf.matmul(X, weights['in']) + biases['in']#shape（128*28,128）
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])#转换为三维shape(128,28,128)
    print("X_in:",X_in)


    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        #forget_bias就是LSTM们的忘记系数，如果等于1，就是不会忘记任何信息。如果等于0，就都忘记
        #state_is_tuple默认就是True，官方建议用True，就是表示返回的状态用一个元祖表示-(steps,batch_size,output_units)。这个里面存在一个状态初始化函数，
        # 就是zero_state（batch_size，dtype）两个参数。batch_size就是输入样本批次的数目，dtype就是数据类型。
    else:
        cell = tf.contrib.rnn.BasicRNNCell(num_units = n_hidden_units)#num_units这个参数的大小就是cell输出结果的维度

    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    #time_major如果是True，就表示RNN的steps用第一个维度表示,如果是False，那么输入的第二个维度就是steps。
    # time_major: If true, these Tensors must be shaped[max_time, batch_size, depth].
    # If false, these Tensors must be shaped [batch_size, max_time, depth]

    #results = tf.matmul(final_state[1], weights['out']) + biases['out']
    #为BasicLSTMcell是可以用这个的，因为他有分线剧情，final state[1] =m_state,=output（-1）

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    predictions = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
    # train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    if not is_training:
        return predictions ,None,None

    loss = tf.losses.mean_squared_error (labels= y,predictions = predictions )

    train_op = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),
                                                   optimizer="Adagrad",learning_rate=0.1)
    return predictions ,loss,train_op


def train(sess,train_x,train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    ds = ds.batch(1 )#当循环次数乘以batch size 大于 数据集的数量，就会报错 end of sequence ，
    # 所以，需要repeat重复使用数据
    x,y = ds.make_one_shot_iterator().get_next()
    print("trainx:", x.shape, "trainy:", y.shape)

    with tf.variable_scope ("model"):
        predictions,loss,train_op = RNN(x,y,weights,biases,True)

    sess.run(tf.global_variables_initializer() )
    for i in range(TRAINING_STEPS ):
        _,l = sess.run([train_op,loss])
        if i % 10 == 0:
            print("train step: " + str(i) +",loss:" + str(l))

def run_eval(sess,test_x,test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(1)#batch(num),num就是一行放的数量，最后一行可能小于num
    x,y = ds.make_one_shot_iterator() .get_next()
    print("testx:",x.shape,"testy:",y.shape)

    with tf.variable_scope("model",reuse=True):#将参数reuse设置为True.这样tf.get_variable会直接获取已经声明的变量，如若变量不存在则会报错；
#reuse为None 或 False 时，tf.get_variable 会创建新的变量，如若同名的变量已经存在，则会报错。
        prediction,_,_= RNN(x,[0.0,0.0],weights,biases,False)

    predictions = []
    labels = []
    for i in range (TEST_EXAMPLES  ):
        p,l = sess.run([prediction,y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions ).squeeze()
    # predictions = np.reshape(predictions,(1,9024),order='F')#C是最后一个维度变化快，二维就是先变列，也就是按行读取
    print("predictions:",predictions.shape)
    labels = np.array(labels ).squeeze()#numpy.squeeze(a,axis = None)把数组中单维度 的去掉，查看数据维度a.shape,删掉单维度后，
    # shape就没有1了
    # labels = np.reshape(labels, (1, 9024), order='F')
    print("labels:",labels.shape)

    rmse = np.sqrt(((predictions - labels )**2).mean(axis=0))#axis=0,输出矩阵是一行，就是求每一列的均值，axis=1，输出一列

    # print("rmse:",rmse.shape)
    #
    # rmse=np.mean(rmse)
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

train_x, train_y = generate_data(flow [0:1248])
print("train_x:",train_x.shape ,"train_y:",train_y.shape)
test_x,test_y = generate_data(flow [1248:1440])
print("test_x:",test_x.shape,"test_y:",test_y.shape )

with tf.Session() as sess:
    train (sess,train_x,train_y)
    run_eval(sess,test_x,test_y)
