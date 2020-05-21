import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import time
import pickle
import math
from openpyxl import load_workbook
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



data = pd.read_excel(r"D:\轨迹预测\2.xlsx", sheet_name='四旋翼')

altitude = data.altitude._values
speed = data.speed._values
flightTime = data.flightTime._values
latitude = data.latitude._values
longitude = data.longitude._values
xz = data.x._values
yz = data.y._values
vy = data.vy._values
v = data.v._values


or_data = np.vstack((altitude, xz, yz, speed, vy)).T

with open(r"D:\轨迹预测\or_data.pkl", 'wb') as f:
    pickle.dump(or_data, f)


start = []
for i in range(1, len(flightTime)):
    if flightTime[i] - flightTime[i-1] < 0 or flightTime[i] - flightTime[i-1] > 100:
        start.append(i)

sq_set = [[] for i in range(len(start) + 1)]

for i in range(0, len(start)-1):
    for j in range(start[i], start[i + 1]):
        sq_set[i+1].append(or_data[j])

for j in range(0, start[0]):
    sq_set[0].append(or_data[j])

for j in range(start[-1], len(flightTime)):
    sq_set[-1].append(or_data[j])

# 以第一个点作为原点，计算坐标
# for i in range(0, len(sq_set)):
#     for j in range(1, len(sq_set[i])):
#         sq_set[i][j][0] = sq_set[i][j][0] - sq_set[i][0][0]
#         sq_set[i][j][1] = sq_set[i][j][1] - sq_set[i][0][1]
#         sq_set[i][j][2] = sq_set[i][j][2] - sq_set[i][0][2]
#
# for i in range(0, len(sq_set)):
#     sq_set[i][0][0] = 0
#     sq_set[i][0][1] = 0
#     sq_set[i][0][2] = 0
#
# sq_set_diff = sq_set

#  以上一个点作为原点，计算坐标
# sq_set_diff = sq_set #不能这样写，diff改变，sq_set也会改变, 也不能用array，因为list长度不一样，怎么办呢

sq_set_diff = [[] for i in range(len(sq_set))]
for i in range(0, len(sq_set)):
    sq_set_diff[i] = [[[] for l in range(5)]for k in range(len(sq_set[i]))]
    sq_set_diff[i][0][0] = 0
    sq_set_diff[i][0][1] = 0
    sq_set_diff[i][0][2] = 0
    sq_set_diff[i][0][3] = 0
    sq_set_diff[i][0][4] = 0
    for j in range(1, len(sq_set[i])):
        sq_set_diff[i][j][0] = sq_set[i][j][0] - sq_set[i][j-1][0]
        sq_set_diff[i][j][1] = sq_set[i][j][1] - sq_set[i][j-1][1]
        sq_set_diff[i][j][2] = sq_set[i][j][2] - sq_set[i][j-1][2]
        sq_set_diff[i][j][3] = sq_set[i][j][3]
        sq_set_diff[i][j][4] = sq_set[i][j][4]



# for i in range(0, len(sq_set_diff)):
#     scalar = MinMaxScaler(feature_range=(0, 1))
#     sq_set_diff[i] = scalar.fit_transform(np.array(sq_set_diff[i])).squeeze()

data_combine = np.array(sq_set_diff[0])
for i in range(1, len(sq_set_diff)):
    data_i = np.array(sq_set_diff[i])
    combine = np.vstack((data_combine, data_i))
    data_combine = combine

scalar = MinMaxScaler(feature_range=(0, 1))
data_combine = scalar.fit_transform(data_combine)

df = pd.DataFrame(data_combine)
df.to_csv(r"D:\轨迹预测\diff_xyz.csv")



time_step = 50
batch_size = 256
hidden_size = 70
# layer_num = 1
output_size = ((1, 3))
class_num = 3


def generate_data(sq):
    x = []
    y = []
    sq = np.array(sq)
    for i in range(0, len(sq)-time_step):
        x.append([sq[i:i + time_step]])
        y.append([sq[i + time_step, [0,1,2]]])
    xs = np.array(x, dtype=np.float32).squeeze()
    ys = np.array(y, dtype=np.float32).squeeze()
    return xs, ys


def lstm_model(x, y,  is_training):

    cell_f = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hidden_size)
                                        for _ in range(layer_num)])

    outputs, _ = tf.nn.dynamic_rnn(cell_f, x, dtype=tf.float32)
    output = outputs[:, -1, :]

    # 回归预测
    nonpredictions = tf.contrib.layers.fully_connected(output, output_size[0] * output_size[1], activation_fn = None )
    predictions = tf.nn.leaky_relu(nonpredictions,alpha = 0.1, name = None)
    if not is_training:
        return predictions, None, None

    predictions = tf.nn.dropout(predictions, 0.95)

    mse = [[] for i in range(output_size[0] * output_size[1])]
    for i in range(0, y.shape[1]):
        a = tf.reduce_mean(tf.square(y[:, i] - predictions[:, i]))
        mse[i].append(a)
    loss = tf.reduce_mean(mse)
    global_step = tf.Variable(0)
    LR = tf.train.exponential_decay(0.1, global_step, int(m / batch_size), 0.96, staircase=True)
    train_op = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),optimizer=opt, learning_rate=LR)

    return  predictions,loss,train_op


def train(sess,x,y):
    ds = tf.data.Dataset.from_tensor_slices((x,y))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    x,y = ds.make_one_shot_iterator().get_next()
    # x = x.eval(session = sess)
    # y = y.eval(session = sess)
    # 传入之前，batchnorm
    # x = tf.layers.batch_normalization(inputs=x, axis=1, training=True)
    # y = tf.layers.batch_normalization(inputs=y, axis=1, training=True)
    with tf.variable_scope("model",reuse=tf.AUTO_REUSE):
        _, train_op, lossor = lstm_model(x, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(train_step):
        _, loss = sess.run([train_op,lossor])

        if i % 100 == 0:
            print('lost: ' + str(i), loss)


def test(sess,x,y, test_step):
    pred_y = np.zeros((test_step * batch_size,  y.shape[1]))
    errorcsv = np.zeros((test_step * batch_size, 3))
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size)
    x, y = ds.make_one_shot_iterator().get_next()
    # x = x.eval(session=sess)
    # y = y.eval(session=sess)
    # 传入之前，batchnorm
    # x = tf.layers.batch_normalization(inputs=x, axis=1, training=False)
    # y = tf.layers.batch_normalization(inputs=y, axis=1, training=False)
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        prediction,_,_ = lstm_model(x, [], False)   # 为了使得测试的时候lstm调用的参数值是训练好的，必须设置reuse=True
    for i in range(test_step):
        pred, ys = sess.run([prediction,y])   # 不能写y = 因为y和prediction被run以后就是ndarray,y这个变量已经存在了，类型是tensor，没法赋值，要能涵盖赋值，必须是同类型
        # for d in range(0, test_y.shape[1]):
        pred_y[i * batch_size:(i + 1) * batch_size, :] = pred

        mse = np.zeros((ys.shape[0],ys.shape[1]))
        mae = np.zeros((ys.shape[0],ys.shape[1]))
        mape = np.zeros((ys.shape[0], ys.shape[1]))
        for k in range(0, ys.shape[1]):
            for l in range(0, ys.shape[0]):
                a = np.square(ys[l, k] - pred[l, k])
                c = np.abs(ys[l, k] - pred[l, k])
                if ys[l, k] <= 0.01 or math.isnan(ys[l, k]):
                    b = 1.0
                else:
                    b = np.abs((ys[l, k] - pred[l, k]) / (ys[l, k]))
                mse[l,k] = a
                mae[l,k] = c
                mape[l,k] = b
        MSE = np.mean(mse,axis = 1)
        MAE = np.mean(mae,axis = 1)
        MAPE = np.mean(mape,axis = 1)
        ERROR = [MSE,MAE,MAPE]
        for e in range(0, 3):
            errorcsv[i*batch_size:(i+1)*batch_size, e] = ERROR[e]

    return pred_y, errorcsv

if __name__=='__main__':

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    MAEscalar = np.zeros((2, 10))

    for t in range(1, 3, 2):
        layer_num = t

        xs, ys = generate_data(data_combine)
        m = len(xs)

        train_end = int(len(xs) * 0.9)
        test_end = int(len(xs))

        if t == 1:
            train_step = 5000
        if t == 3:
            train_step = 8000
        if t == 5:
            train_step = 5000
        if t == 7:
            train_step = 6000
        if t == 9:
            train_step = 6000

        test_step = (test_end - train_end) // batch_size

        train_x = xs[0:train_end]
        train_y = ys[0:train_end]
        test_x = xs[train_end:test_end]
        test_y = ys[train_end:test_end]

        ceshiji = np.reshape(test_x, [-1, 5])
        dt1 = pd.DataFrame(ceshiji)
        dt1.to_csv(r"D:\轨迹预测" + '\\' + '测试集x' + str(time_step) + ".csv",)
        dt2 = pd.DataFrame(test_y)
        dt2.to_csv(r"D:\轨迹预测" + '\\' + '测试集y' + str(time_step) + ".csv",)
        root = r"D:\轨迹预测\prediction"
        sub_set = ["LSTM"]
        for s in range(0, 1):
            sub = sub_set[s]
            optimizer = ["Adam"]
            for p in range(0, len(optimizer)):
                opt = optimizer[p]
                b = os.path.exists(root + "\\" + sub + "\\" + opt)
                if b:
                    print("path exist")
                else:
                    os.makedirs(root + "\\" + sub + "\\" + opt)

                time_start = time.time()
                train(sess, train_x, train_y)
                pred, errorcsv = test(sess, test_x, test_y, test_step)
                dt = pd.DataFrame(errorcsv)
                dt.to_csv(root + "\\" + sub + "\\" + "error5_50.csv")
                time_end = time.time()
                MAEscalar[1, t] = time_end-time_start

                MAEscalar[0, t] = np.mean(errorcsv[:, 1])

            dt = pd.DataFrame(MAEscalar)
            dt.to_csv(root + "\\" + sub + "\\" + "MAE5_50.csv")



