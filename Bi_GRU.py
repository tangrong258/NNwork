import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
import time
import pickle
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


Or_data = np.vstack((altitude, xz, yz, speed, vy)).T


def separate_track(or_data):
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

    return sq_set


def generate_data(sq, time_step):
    x = []
    y = []
    sq = np.array(sq)
    for i in range(0, len(sq)-time_step):
        x.append([sq[i:i + time_step]])
        y.append([sq[i + time_step, [0,1,2]]])  # 对于这个网络
    xs = np.array(x, dtype=np.float32).squeeze()
    ys = np.array(y, dtype=np.float32).squeeze()
    return xs, ys


def get_minlen_sample(sq_set, sq_len):
    sam = []
    y = []
    for i in range(len(sq_set)):

        if len(sq_set[i]) > sq_len + 1:  # 希望是个二维的，一维就是一条数据，不利于后面stack, 而且相减，所以至少长度为3
            for j in range(1, len(sq_set[i])):
                sq_set[i][j][0] = sq_set[i][j][0] - sq_set[i][0][0]
                sq_set[i][j][1] = sq_set[i][j][1] - sq_set[i][0][1]
                sq_set[i][j][2] = sq_set[i][j][2] - sq_set[i][0][2]

            sq_set[i][0][0] = 0
            sq_set[i][0][1] = 0
            sq_set[i][0][2] = 0

            x_s, y_s = generate_data(sq_set[i], sq_len)
            sam.append(x_s)
            y.append(y_s)

    sam_len = np.array(sam[0])
    y_len = np.array(y[0])

    for j in range(1, len(sam)):
        sam_len = np.vstack((sam_len, np.array(sam[j])))
        y_len = np.vstack((y_len, np.array(y[j])))

    return sam_len, y_len



def get_diff_time_seq(ts, sample):
    x = [[] for _ in range(6)]
    x_array = []

    def diff_6(ser):
        dt = ts + 1
        x_10 = []
        for i in range(ts+1):
            x_10.append(ser[len(ser)-1-i*dt] - ser[len(ser)-1-(i+1)*dt])

        x_10 = np.array(x_10)
        return x_10[::-1]  # np.flipud()是将行反向存储， np.fliplr()是将列反向存储，
                           # opencv.flip(m,axis)记住图像的以一个维度是宽度，刚好是列，
                           # 所以aixs数值和矩阵是反过来的

    def diff_5(ser):
        dt = ts
        x_9 = []
        for i in range(ts+1):
            x_9.append(ser[len(ser)-2-i*dt] - ser[len(ser)-2-(i+1)*dt])

        x_9 = np.array(x_9)
        return x_9[::-1]

    def diff_4(ser):
        dt = ts-1
        x_8 = []
        for i in range(ts+1):
            x_8.append(ser[len(ser)-3-i*dt] - ser[len(ser)-3-(i+1)*dt])

        x_8 = np.array(x_8)
        return x_8[::-1]

    def diff_3(ser):
        dt = ts-2
        x_7 = []
        for i in range(ts+1):
            x_7.append(ser[len(ser)-4-i*dt] - ser[len(ser)-4-(i+1)*dt])

        x_7 = np.array(x_7)
        return x_7[::-1]

    def diff_2(ser):
        dt = ts-3
        x_6 = []
        for i in range(ts+1):
            x_6.append(ser[len(ser)-5-i*dt] - ser[len(ser)-5-(i+1)*dt])

        x_6 = np.array(x_6)
        return x_6[::-1]

    def diff_1(ser):
        dt = ts-4
        x_5 = []
        for i in range(ts+1):
            x_5.append(ser[len(ser)-6-i*dt] - ser[len(ser)-6-(i+1)*dt])

        x_5 = np.array(x_5)
        return x_5[::-1]

    for i in range(sample.shape[0]):
        Ser = sample[i]
        x[5].append(diff_6(Ser))
        x[4].append(diff_5(Ser))
        x[3].append(diff_4(Ser))
        x[2].append(diff_3(Ser))
        x[1].append(diff_2(Ser))
        x[0].append(diff_1(Ser))

    for k in range(len(x)):
        x_array.append(np.array(x[k]))

    return x_array

def sample_sca(x):
    # 一定是在相减之前进行归一化，不然就预测差值以后无法返回求具体的数值，Sample.shape=[n,Min_len,input_size]
    x_2d = np.reshape(x, [-1, x.shape[2]])
    sca_ler = MinMaxScaler(feature_range=(0, 1))
    x_2d_sca = sca_ler.fit_transform(x_2d)
    x_sca = np.reshape(x_2d_sca, [x.shape[0], x.shape[1],x.shape[2]])

    return x_sca, sca_ler


def diff_sca(x):
    sca_set = []
    x_set = []
    for i in range(len(x)):
        x_2d = np.reshape(x[i], [-1, x[i].shape[1]])
        sca = MinMaxScaler(feature_range=(0, 1))
        x_2d_sca = sca.fit_transform(x_2d)
        x_sca = np.reshape(x_2d_sca, [-1, x[i].shape[1], x[i].shape[2]])
        x_set.append(x_sca)
        sca_set.append(sca)
    return np.array(x_set), sca_set


keep_prob = 0.95
batch_size = 100
# hidden_size = 5
layer_num = 2
output_size = ((1, 3))
Ts = 5
min_len = int((Ts+1) * (Ts+1) + 1)


def weight_variable(shape, name):
    initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0, )
    return tf.get_variable(shape=shape, initializer=initializer, name=name)


def bias_variable(shape, name):
    initializer = tf.constant_initializer(0.1)
    return tf.get_variable(shape=shape, initializer=initializer, name=name)


def nn_work(x0, x1, x2, x3, x4, x5, y,  base, target, is_training):

    #  先计算间隔为1-6的数据，得到预测值未来的值
    cell_1 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(layer_num)])
    outputs_1, _ = tf.nn.dynamic_rnn(cell_1, x1, dtype=tf.float32)
    output_1 = outputs_1[:, -1, :]
    predictions_1 = tf.contrib.layers.fully_connected(output_1, Base_point.shape[1], activation_fn = None )
    predictions_1 = tf.nn.leaky_relu(predictions_1, alpha=0.1, name=None)
    n_1 = base + predictions_1

    cell_2 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(layer_num)])
    outputs_2, _ = tf.nn.dynamic_rnn(cell_2, x2, dtype=tf.float32)
    output_2 = outputs_2[:, -1, :]
    predictions_2 = tf.contrib.layers.fully_connected(output_2, Base_point.shape[1], activation_fn = None )
    predictions_2 = tf.nn.leaky_relu(predictions_2, alpha=0.1, name=None)
    n_2 = base + predictions_2

    cell_3 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(layer_num)])
    outputs_3, _ = tf.nn.dynamic_rnn(cell_3, x3, dtype=tf.float32)
    output_3 = outputs_3[:, -1, :]
    predictions_3 = tf.contrib.layers.fully_connected(output_3, Base_point.shape[1], activation_fn = None )
    predictions_3 = tf.nn.leaky_relu(predictions_3, alpha=0.1, name=None)
    n_3 = base + predictions_3

    cell_4 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(layer_num)])
    outputs_4, _ = tf.nn.dynamic_rnn(cell_4, x4, dtype=tf.float32)
    output_4 = outputs_4[:, -1, :]
    predictions_4 = tf.contrib.layers.fully_connected(output_4, Base_point.shape[1], activation_fn = None )
    predictions_4 = tf.nn.leaky_relu(predictions_4, alpha=0.1, name=None)
    n_4 = base + predictions_4

    cell_5 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(layer_num)])
    outputs_5, _ = tf.nn.dynamic_rnn(cell_5, x5, dtype=tf.float32)
    output_5 = outputs_5[:, -1, :]
    predictions_5 = tf.contrib.layers.fully_connected(output_5, Base_point.shape[1], activation_fn = None )
    predictions_5 = tf.nn.leaky_relu(predictions_5, alpha=0.1, name=None)
    n_5 = base + predictions_5

    # 使用n_1_5预测n, 先构造n_1 到 n_5 的输入序列,tf.concat()
    if len(n_5.shape) < 3:
        n_5 = tf.expand_dims(n_5, axis=1)
        n_4 = tf.expand_dims(n_4, axis=1)
        n_3 = tf.expand_dims(n_3, axis=1)
        n_2 = tf.expand_dims(n_2, axis=1)
        n_1 = tf.expand_dims(n_1, axis=1)

    x6_input = n_5
    n_seq = [n_4, n_3, n_2, n_1]
    for i in range(0, len(n_seq)):
        x6_input = tf.concat([x6_input, n_seq[i]], 1)

    cell_6 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(layer_num)])
    outputs_6, _ = tf.nn.dynamic_rnn(cell_6, x6_input, dtype=tf.float32)
    output_6 = outputs_6[:, -1, :]
    predictions_6 = tf.contrib.layers.fully_connected(output_6, Target_point.shape[1], activation_fn = None )
    predictions_6 = tf.nn.leaky_relu(predictions_6, alpha=0.1, name=None)  # 这里已经是反向序列真实预测到target，而不是差值

    # 计算最基础的预测序列
    cell_0 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(layer_num)])
    outputs_0, _ = tf.nn.dynamic_rnn(cell_0, x0, dtype=tf.float32)
    output_0 = outputs_0[:, -1, :]
    predictions_0 = tf.contrib.layers.fully_connected(output_0, Target_point.shape[1], activation_fn=None)
    predictions_0 = tf.nn.leaky_relu(predictions_0, alpha=0.1, name=None)
    n_0 = base + predictions_0  # n_0是间隔为1正向序列得预测值，是加上了差值得

    # 至此有两种预测值正向的n_0,反向的prediction_6,

    W1 = weight_variable([Target_point.shape[1]], 'W1')
    W2 = weight_variable([Target_point.shape[1]], 'W2')
    B = bias_variable([batch_size, 1], 'B')
    predictions = tf.multiply(W1, n_0) + tf.multiply(W2, predictions_6) + B

    if not is_training:
        return predictions, None, None

    predictions = tf.nn.dropout(predictions, rate=1-keep_prob)

    mse = [[] for _ in range(output_size[1])]

    for i in range(0, output_size[1]):
        a = tf.reduce_mean(tf.square(target[:, i] - predictions[:, i]))
        mse[i].append(a)

    loss = tf.reduce_mean(mse)
    global_step = tf.Variable(0)
    LR = tf.train.exponential_decay(0.1, global_step, int(m / batch_size), 0.96, staircase=True)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer=opt, learning_rate=LR)
    return predictions, loss, train_op


def train(x0, x1, x2, x3, x4, x5, y, base, target):
    ds = tf.data.Dataset.from_tensor_slices((x0, x1, x2, x3, x4, x5, y, base, target))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    x0, x1, x2, x3, x4, x5, y, base, target = ds.make_one_shot_iterator().get_next()
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        pred, train_op, lossor = nn_work(x0, x1, x2, x3, x4, x5, y, base, target, True)

    sess.run(tf.global_variables_initializer())
    train_loss = np.zeros(train_step)
    train_time = np.zeros(train_step)
    time_line = 0
    for i in range(train_step):
        time_s = time.time()
        pred1, _,  loss = sess.run([pred, train_op, lossor])
        time_e = time.time()
        time_line = time_line + (time_e - time_s)
        train_time[i] = time_line
        train_loss[i] = loss
        if i % 100 == 0:
            print('lost: ' + str(i), loss)

    return train_time, train_loss


def test(x0, x1, x2, x3, x4, x5, y, base, target, test_step):
    pred_y = np.zeros((test_step * batch_size,  y.shape[1]))
    errorcsv = np.zeros((test_step * batch_size, 3))
    ds = tf.data.Dataset.from_tensor_slices((x0, x1, x2, x3, x4, x5, y, base, target))
    ds = ds.batch(batch_size)
    x0, x1, x2, x3, x4, x5, y, base, target = ds.make_one_shot_iterator().get_next()
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        prediction,_,_ = nn_work(x0, x1, x2, x3, x4, x5, [], base, [], False)  # 为了使得测试的时候lstm调用的参数值是训练好的，必须设置reuse=True


    for i in range(test_step):
        pred, ys = sess.run([prediction, y])   # 不能写y = 因为y和prediction被run以后就是ndarray,y这个变量已经存在了，类型是tensor，没法赋值，要能涵盖赋值，必须是同类型
        pred_y[i * batch_size:(i + 1) * batch_size, :] = pred

        mse = np.zeros((ys.shape[0], ys.shape[1]))
        mae = np.zeros((ys.shape[0],ys.shape[1]))
        mape = np.zeros((ys.shape[0], ys.shape[1]))
        for k in range(0, ys.shape[1]):
            for l in range(0, ys.shape[0]):
                a = np.square(ys[l, k] - pred[l, k])
                c = np.abs(ys[l, k] - pred[l, k])
                b = np.abs((ys[l, k] - pred[l, k]) / (ys[l, k]))
                mse[l,k] = a
                mae[l,k] = c
                mape[l,k] = b
        MSE = np.mean(mse,axis = 1)
        MAE = np.mean(mae,axis = 1)
        MAPE = np.mean(mape,axis = 1)
        ERROR = [MSE, MAE, MAPE]

        for e in range(0, 3):
            errorcsv[i*batch_size:(i+1)*batch_size, e] = ERROR[e]

    return pred_y, errorcsv


if __name__=='__main__':
    root = r"D:\轨迹预测\prediction"
    sub = "GRU"
    opt = "Adagrad"
    b = os.path.exists(root + "\\" + sub + "\\" + opt)
    if b:
        print("path exist")
    else:
        os.makedirs(root + "\\" + sub + "\\" + opt)

    no_name = [[] for _ in range(1)]
    MAE_avg = []
    for bb in range(1):
        MAEscalar = np.zeros((6, 70))
        train_mse = []
        for t in range(0, 1, 1):
            tf.reset_default_graph()
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            hidden_size = (t + 1) * 5

            Sq_set = separate_track(Or_data)
            Sample, _ = get_minlen_sample(Sq_set, min_len)  # Sample.shape=[n,Min_len,input_size]
            Sample_Sca, Sca_ler = sample_sca(Sample)
            Base_point = Sample_Sca[:, min_len-(Ts + 1), :]
            Target_point = Sample_Sca[:, min_len - Ts, :]
            X = get_diff_time_seq(Ts, Sample_Sca)  # X[0].shape = [n, Ts+1, input_size]
            X_4D = np.array(X)

            # Sample, _, Sq_set_diff = get_minlen_sample(Sq_set, min_len)  # Sample.shape=[n,Min_len,input_size]
            # Base_point = Sample[:, min_len - (Ts + 1), :]
            # Target_point = Sample[:, min_len - Ts, :]
            # X = get_diff_time_seq(Ts, Sample)  # X[0].shape = [n, Ts+1, input_size]
            # X_4D, Sca_set = diff_sca(X)

            m = len(Sample)
            train_end = int(len(Sample) * 0.75)
            test_end = int(len(Sample))
            train_step = 3000
            test_step = (test_end - train_end) // batch_size

            X_train = X_4D[:, 0:train_end, 0:-1, :]
            Y_train = X_4D[:, 0:train_end, -1, :]
            X_test = X_4D[:, train_end:test_end, 0:-1, :]
            Y_test = X_4D[:, train_end:test_end, -1, :]

            time_start = time.time()

            time_train, loss_train = train(X_train[0], X_train[1], X_train[2], X_train[3], X_train[4], X_train[5],
                                           Y_train[0], Base_point[0:train_end, :], Target_point[0:train_end, :])

            train_mse.append([time_train, loss_train])

            pred, errorcsv = test(X_test[0], X_test[1], X_test[2], X_test[3], X_test[4], X_test[5], Y_test[0],
                                           Base_point[train_end:test_end, :], Target_point[train_end:test_end, :], Ts)

            time_end = time.time()
            MAEscalar[1, t] = time_end-time_start
            MAEscalar[0, t] = np.mean(errorcsv[:, 1])


        for i in range(len(train_mse)):
            no_name[i].append(train_mse[i])
            MAE_avg.append(MAEscalar)
            # dt = pd.DataFrame(np.array(train_mse[i]).T)
            # dt.to_csv(root + "\\" + sub + "\\" + "train_loss" + str(i) + '_' + str(bb) + ".csv")

        # dt = pd.DataFrame(MAEscalar)
        # dt.to_csv(root + "\\" + sub + "\\" + "MAE5_50.csv")

    for tt in range(len(no_name)):
        To_excel = np.mean(np.array(no_name[tt]), 0)
        MAE_toexcel = np.mean(np.array(MAE_avg), 0)
        dt = pd.DataFrame(To_excel.T)
        dt.to_csv(root + "\\" + sub + "\\" + "train_loss" + '_mean' + str(tt) + ".csv")
        dt = pd.DataFrame(MAE_toexcel)
        dt.to_csv(root + "\\" + sub + "\\" + "MAE" + '_mean' + ".csv")
