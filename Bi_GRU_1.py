import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
import time
from Bi_GRU import separate_track, generate_data


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


def get_minlen_sample(sq_set, sq_len):
    sam = []
    y = []
    sq_set_diff = [[] for _ in range(len(sq_set))]

    for i in range(len(sq_set)):

        if len(sq_set[i]) > sq_len + 1:  # 希望是个二维的，一维就是一条数据，不利于后面stack, 而且相减，所以至少长度为3
            for j in range(1, len(sq_set[i])):
                sq_set[i][j][0] = sq_set[i][j][0] - sq_set[i][0][0]
                sq_set[i][j][1] = sq_set[i][j][1] - sq_set[i][0][1]
                sq_set[i][j][2] = sq_set[i][j][2] - sq_set[i][0][2]

            sq_set[i][0][0] = 0
            sq_set[i][0][1] = 0
            sq_set[i][0][2] = 0

            scalar = MinMaxScaler(feature_range=(0, 1))
            sq_set_diff[i] = scalar.fit_transform(np.array(sq_set[i])).squeeze()

            x_s, y_s = generate_data(sq_set_diff[i], sq_len)
            sam.append(x_s)
            y.append(y_s)

    sam_len = np.array(sam[0])
    y_len = np.array(y[0])

    for j in range(1, len(sam)):
        sam_len = np.vstack((sam_len, np.array(sam[j])))
        y_len = np.vstack((y_len, np.array(y[j])))

    return sam_len, y_len, sq_set_diff


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
        for i in range(int(30 / 2)):
            x_6.append(ser[len(ser)-5-i*dt] - ser[len(ser)-5-(i+1)*dt])

        x_6 = np.array(x_6)
        return x_6[::-1]

    def diff_1(ser):
        dt = ts-4
        x_5 = []
        for i in range(int(30 / 1)):
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


def diff_sca(x):
    sca_set = []
    x_set = []
    for i in range(len(x)):
        x_2d = np.reshape(x[i], [-1, x[i].shape[2]])
        sca = MinMaxScaler(feature_range=(0, 1))
        x_2d_sca = sca.fit_transform(x_2d)
        x_sca = np.reshape(x_2d_sca, [-1, x[i].shape[1], x[i].shape[2]])
        x_set.append(x_sca)
        sca_set.append(sca)
    return x_set, sca_set


def fit_scaler(scaler2, x, x_is_diff, scope):
    sca_fit = scaler2
    if not x_is_diff:
        diff_x = np.zeros(((x.shape[0], x.shape[1]-1, x.shape[2])), dtype='float32')
        for i in range(diff_x.shape[1]):
            diff_x[:, i, :] = (x[:, i+1, :] - x[:, 0, :]) / (i + 1)

    else:
        diff_x = np.zeros(((x.shape[0], x.shape[1], x.shape[2])), dtype='float32')

        diff_x = x / scope

    diff_x_2d = np.reshape(diff_x, [-1, diff_x.shape[2]])
    diff_x_sca_2d = sca_fit.transform(diff_x_2d)
    diff_x_sca = np.reshape(diff_x_sca_2d, [-1, diff_x.shape[1], diff_x.shape[2]])

    return diff_x_sca


index = 0
scope = 2
keep_prob = 0.95
optimizer = "Adagrad"
batch_size = 100
# hidden_size = 5
layer_num = 2
output_size = 3
Ts = 5
min_len = int((Ts+1) * (Ts+1) + 1)

def weight_variable(shape, name):
    initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0, )
    return tf.get_variable(shape=shape, initializer=initializer, name=name)


def bias_variable(shape, name):
    initializer = tf.constant_initializer(0.1)
    return tf.get_variable(shape=shape, initializer=initializer, name=name)


def compute_loss(y, prediction, out_put_size):
    mse = [[] for i in range(out_put_size)]
    for i in range(0, out_put_size):
        a = tf.reduce_mean(tf.square(y[:, i] - prediction[:, i]))
        mse[i].append(a)

    loss = tf.reduce_mean(mse)
    global_step = tf.Variable(0)
    LR = tf.train.exponential_decay(0.1, global_step, int(m / batch_size), 0.96, staircase=True)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer=optimizer, learning_rate=LR)
    return loss, train_op



class Fpmodel(object):

    def __init__(self, nn_input, out_put_size, train_bool):

        self.nn_input = nn_input
        self.input_x = self.nn_input[:, 0:-1, :]
        self.input_y = self.nn_input[:, -1, :]
        self.out_put_size = out_put_size
        self.train_bool = train_bool

        with tf.variable_scope('diff_1_nn'):
            self.predictions, self.loss, self.train_op = self.diff_nn(
                self.input_x, self.input_y, self.out_put_size, self.train_bool)

    def GRU_model(self, x, out_put_size):
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(hidden_size)
                                              for _ in range(layer_num)])
        outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        output = outputs[:, -1, :]

        predictions = tf.contrib.layers.fully_connected(output, out_put_size, activation_fn=None)
        predictions = tf.nn.leaky_relu(predictions, alpha=0.1, name=None)
        predictions = tf.nn.dropout(predictions, rate=1-keep_prob)

        return predictions

    def diff_nn(self, x, y, out_put_size, is_training):
        predictions = self.GRU_model(x, out_put_size)
        if not is_training:
            return predictions, None, None

        loss, train_op = compute_loss(y, predictions, out_put_size)
        return predictions, loss, train_op


class BehFront(object):

    def __init__(self, y, diff, prediction, out_put_size, train_bool):  # diff 的第一个步长就是y
        self.y = y
        self.diff = diff
        self.prediction = prediction
        self.out_put_size = out_put_size
        self.train_bool = train_bool

        self.W1 = weight_variable([self.out_put_size], 'W_prediction')
        self.W2 = weight_variable([self.out_put_size], 'W_diff_geo')
        self.Bias = bias_variable([batch_size, 1], 'Bias')

        with tf.variable_scope('diff_1_nn'):
            self.predictions, self.loss, self.train_op = self.improve(self.train_bool)

    def weight_model(self):
        predictions = tf.multiply(self.W1, self.prediction) + tf.multiply(self.W2, self.diff) + self.Bias
        return predictions

    def improve(self, is_training):
        predictions = self.weight_model()

        if not is_training:
            return predictions, None, None

        loss, train_op = compute_loss(self.y, predictions, self.out_put_size)

        return predictions, loss, train_op


def train(x1, diff, x2, outsize, sca_inv, sca_fit):
    ds = tf.data.Dataset.from_tensor_slices((x1, diff, x2))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    x1, diff, x2 = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model1", reuse=tf.AUTO_REUSE):
        training1 = Fpmodel(x1, outsize, True)
        pred1, train_op1, lossor1 = training1.predictions, training1.train_op, training1.loss

    with tf.variable_scope("model_1", reuse=tf.AUTO_REUSE):
        training_1 = Fpmodel(x2, 5, True)
        pred_1, train_op_1, lossor_1 = training_1.predictions, training_1.train_op, training_1.loss

        sess.run(tf.global_variables_initializer())
        pred_1_sca, _,  _ = sess.run([pred_1, train_op_1, lossor_1])
        pred_1_inv = sca_inv.inverse_transform(pred_1_sca)
        pred_1_fit = sca_fit.transform(pred_1_inv / 2)


    with tf.variable_scope("model_comb", reuse=tf.AUTO_REUSE):

        training2 = BehFront(x1[:, -1, :], pred_1_fit[:, 0:outsize], pred1, outsize, True)
        # diff_array = sess.run(diff)
        # training2 = BehFront(x1[:, -1, :], diff, pred1, outsize, True)
        pred2, train_op2, lossor2 = training2.predictions, training2.train_op, training2.loss


    sess.run(tf.global_variables_initializer())

    train_loss = np.zeros(train_step)
    train_time = np.zeros(train_step)
    time_line = 0
    for i in range(train_step):
        time_s = time.time()
        _, _,  loss1, _, _, loss2 = sess.run([pred1, train_op1, lossor1, pred2, train_op2, lossor2])

        loss = loss2

        time_e = time.time()
        time_line = time_line + (time_e - time_s)
        train_time[i] = time_line
        train_loss[i] = loss
        if i % 100 == 0:
            print('lost1: ' + str(i), loss1, 'loss2:', loss2)

    return train_time, train_loss


def test(x1, diff, x2, outsize, sca_inv, sca_fit, test_step):
    pred_y = np.zeros((test_step * batch_size,  outsize))
    errorcsv = np.zeros((test_step * batch_size, 3))
    ds = tf.data.Dataset.from_tensor_slices((x1, diff, x2))
    ds = ds.batch(batch_size)
    x1, diff, x2 = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model1", reuse=tf.AUTO_REUSE):
        testing1 = Fpmodel(x1, outsize, False)
        prediction1 = testing1.predictions  # 为了使得测试的时候lstm调用的参数值是训练好的，必须设置reuse=True

    with tf.variable_scope("model_1", reuse=tf.AUTO_REUSE):
        testing_1 = Fpmodel(x2, 5, False)
        prediction_1 = testing_1.predictions

        pred_1_inv = sca_inv.inverse_transform(sess.run(prediction_1))
        pred_1_fit = sca_fit.transform(pred_1_inv / 2)


    with tf.variable_scope("model_comb", reuse=tf.AUTO_REUSE):
        testing2 = BehFront(x1[:, -1, :], pred_1_fit[:, 0:outsize], prediction1, outsize, False)
        # testing2 = BehFront(x1[:, -1, :], diff, prediction1, outsize, False)
        prediction2 = testing2.predictions


    for i in range(test_step):
        pred_1, ys, pred_2 = sess.run([prediction1, x1[:, -1, :], prediction2])   # 不能写y = 因为y和prediction被run以后就是ndarray,y这个变量已经存在了，类型是tensor，没法赋值，要能涵盖赋值，必须是同类型

        pred = pred_2

        pred_y[i * batch_size:(i + 1) * batch_size, :] = pred

        mse = np.zeros((pred.shape[0], pred.shape[1]))
        mae = np.zeros((pred.shape[0], pred.shape[1]))
        mape = np.zeros((pred.shape[0], pred.shape[1]))
        for k in range(0, pred.shape[1]):
            for l in range(0, pred.shape[0]):
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
            Sample, _, Sq_set_diff = get_minlen_sample(Sq_set, min_len)  # Sample.shape=[n,Min_len,input_size]

            Base_point = Sample[:, min_len-(Ts + 1):min_len-(Ts-2), :]

            X = get_diff_time_seq(Ts, Sample)  # X[0].shape = [n, Ts+1, input_size]
            X_list, Sca_set = diff_sca(X)

            Diff_sca = fit_scaler(Sca_set[0], Base_point, False, scope)
            Diff_sca1 = fit_scaler(Sca_set[0], X[1], True, scope)

            m = len(Sample)
            train_end = int(len(Sample) * 0.75)
            test_end = int(len(Sample))
            train_step = 6000
            test_step = (test_end - train_end) // batch_size

            train_values1 = X_list[index][0:train_end, :, :]
            test_values1 = X_list[index][train_end:test_end, :, :]

            train_values_1 = X_list[1][0:train_end, :, :]
            test_values_1 = X_list[1][train_end:test_end, :, :]

            train_diff_sca = Diff_sca[0:train_end, -1, 0:output_size]
            test_diff_sca = Diff_sca[train_end:test_end, -1, 0:output_size]

            time_start = time.time()

            time_train, loss_train = train(train_values1, train_diff_sca, train_values_1, output_size, Sca_set[1], Sca_set[0])
            train_mse.append([time_train, loss_train])
            pred, errorcsv = test(test_values1, test_diff_sca, test_values_1, output_size, Sca_set[1], Sca_set[0], test_step -1)


            time_end = time.time()
            MAEscalar[1, t] = time_end-time_start
            MAEscalar[0, t] = np.mean(errorcsv[:, 1])

        for i in range(len(train_mse)):
            no_name[i].append(train_mse[i])
            MAE_avg.append(MAEscalar)


    for tt in range(len(no_name)):
        To_excel = np.mean(np.array(no_name[tt]), 0)
        MAE_toexcel = np.mean(np.array(MAE_avg), 0)
        dt = pd.DataFrame(To_excel.T)
        dt.to_csv(root + "\\" + sub + "\\" + "train_loss" + '_mean' + str(tt) + ".csv")
        dt = pd.DataFrame(MAE_toexcel)
        dt.to_csv(root + "\\" + sub + "\\" + "MAE" + '_mean' + ".csv")