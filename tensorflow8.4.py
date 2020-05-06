import tensorflow as tf
import numpy as np
import csv
from tkinter import _flatten
import pandas as pd

import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib import pyplot as plt

csvFile = open(r'C:\Users\tangrong\Desktop\ZSNJ201805.csv', "r")
reader = csv.reader(csvFile)
list1=[]
list2=[]
#全部读进来，再操作
# print(reader)
for row in reader:
    #list1 += [row[1],row[2],row[3]]
    list2 += [row[1]]#它迭代的读取貌似是一个一个数据读

#list1 = np.array(list1).reshape(len(list1)//3,3)
#list2 = np.array(list2).reshape(len(list2),1)
delaytime = list(_flatten(list2))
delaytime = np.array(delaytime,dtype= np.float32)
flow = []
for x in delaytime:
    x = float(x - np.min(delaytime))/(np.max(delaytime )- np.min(delaytime))
    flow.append(x)
# print("delaytime:", flow)


HIDDEN_SIZE = 900
NUM_LAYERS = 1

TIME_STEPS = 4
TRAINING_STEPS =1000 #1248-TIME_STEPS
BATCH_SIZE =30#batch 是计算一次cost的集合，数据量小的时候，使用一个batch即可，

TRAINING_EXAMPLES = 1244
TEST_EXAMPLES = 192-TIME_STEPS
SAMPLE_GAP = 1


def generate_data(seq):
    x = []
    y = []
    for i in range (len(seq)-TIME_STEPS ):
        x.append([seq[i:i+TIME_STEPS ]])
        y.append([seq[i+TIME_STEPS ]])
    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)

def add_input_layer(self,):
    l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)

    Ws_in = self._weight_variable([self.input_size, self.cell_size])

    bs_in = self._bias_variable([self.cell_size,])

    with tf.name_scope('Wx_plus_b'):
        l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in

    self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
def lstm_model(x,y,is_training):

    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell (HIDDEN_SIZE)
                                        for _ in range(NUM_LAYERS)])
    outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    output = outputs[:,-1,:]

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
    # 所以，需要repeat重复使用数据，repeat().shuffle(1000)
    x,y = ds.make_one_shot_iterator().get_next()
    print("trainx:", x.shape, "trainy:", y.shape)

    with tf.variable_scope ("model"):
        predictions,loss,train_op = lstm_model(x,y,True)

    sess.run(tf.global_variables_initializer() )
    for i in range(TRAINING_STEPS ):
        _,l = sess.run([train_op,loss])
        if i % 100 == 0:
            print("train step: " + str(i) +",loss:" + str(l))
    print("loss:",loss.shape)

def run_eval(sess,test_x,test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(1 )
    x,y = ds.make_one_shot_iterator() .get_next()
    print("testx:", x.shape, "testy:", y.shape)

    with tf.variable_scope("model",reuse=True):
        prediction,_,_= lstm_model(x,[0.0],False)#prediction是个tensor，shape（？，1）

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

    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'predictions': predictions, 'labels': labels})  # 两种写法
    # columns = ["URL", "predict", "score"]
    # dt = pd.DataFrame(result_list, columns=columns),直接创建
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(r"C:\Users\tangrong\Desktop\delaytimetest.csv", index=False, sep=',')

    plt.plot(predictions ,label = 'predictions')
    plt.plot(labels,label = 'real')
    plt.show()

# test_start = delaytime (TRAINING_EXAMPLES + TIME_STEPS )* SAMPLE_GAP
# test_end = test_start + (TEST_EXAMPLES  + TIME_STEPS )* SAMPLE_GAP
#
# train_x, train_y = generate_data(np.sin(np.linspace(0,test_start,TRAINING_EXAMPLES + TIME_STEPS,
#                                                     dtype=np.float32)))
# test_x,test_y = generate_data(np.sin(np.linspace(test_start ,test_end ,TEST_EXAMPLES  + TIME_STEPS,
#                                                  dtype=np.float32)))

train_x, train_y = generate_data(flow [0:1248])
print("train_x:",train_x.shape)
test_x,test_y = generate_data(flow [1248:1440])
print("test_x:",test_x.shape )
# [1,2]的shape值(2,)，意思是一维数组，数组中有2个元素。
#
# [[1],[2]]的shape值是(2,1)，意思是一个二维数组，每行有1个元素。所以shape为（188,2,4）三维，每行有四个元素，一个二维是2行，一共有188个二维
#
# [[1,2]]的shape值是（1，2），意思是一个二维数组，每行有2个元素。
with tf.Session() as sess:
    train (sess,train_x,train_y)
    run_eval(sess,test_x,test_y)

