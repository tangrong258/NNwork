import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

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
for i in range(0, len(sq_set)):
    for j in range(1, len(sq_set[i])):
        sq_set[i][j][0] = sq_set[i][j][0] - sq_set[i][0][0]
        sq_set[i][j][1] = sq_set[i][j][1] - sq_set[i][0][1]
        sq_set[i][j][2] = sq_set[i][j][2] - sq_set[i][0][2]

for i in range(0, len(sq_set)):
    sq_set[i][0][0] = 0
    sq_set[i][0][1] = 0
    sq_set[i][0][2] = 0

#前面处理为每一个轨迹为一个列表，第一个点为原点
#下方筛选连续低俗，高速的点，通过generate函数进行筛选


# sq_setscalar = sq_set
sq_setscalar = [[] for i in range(len(start) + 1)]#这里不能使用列表直接相同去操作，归一化会一起改变

for i in range(0, len(sq_set)):
    scalar = MinMaxScaler(feature_range=(0, 1))
    sq_setscalar[i] = scalar.fit_transform(sq_set[i]).squeeze()



time_step = 5
batch_size = 20
hidden_size = 50
layer_num = 2
output_size = ((1, 1))

def generate_data(sq, sq_org):
    xo = []
    yo = []
    xu = []
    yu = []
    sq = np.array(sq)#必须是一个array
    sq_org = np.array(sq_org)
    for i in range(0, len(sq) - time_step):
        if np.max(sq_org[i:i + time_step, 3]) < 5.0:
            xo.append(sq[i:i + time_step])
            yo.append(sq[i + time_step, [0, 1, 2]])

        if np.min(sq_org[i:i + time_step, 3]) >= 5.0:
            xu.append(sq[i:i + time_step])
            yu.append(sq[i + time_step, [0, 1, 2]])

    xso = np.array(xo, dtype=np.float32)
    yso = np.array(yo, dtype=np.float32)

    xsu = np.array(xu, dtype=np.float32)
    ysu = np.array(yu, dtype=np.float32)
    return xso, yso, xsu, ysu


def lstm_model(x, y,  is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(hidden_size)
                                        for _ in range(layer_num)])
    outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    output = outputs[:, -1, :] #[batchsize,hiddensize]

    nonpredictions = tf.contrib.layers.fully_connected(output, output_size[0] * output_size[1], activation_fn = None )
    predictions = tf.nn.leaky_relu(nonpredictions,alpha = 0.2, name = None)
    if not is_training:
        return predictions ,None,None

    # predictions = tf.nn.dropout(predictions,0.5)
    mse = [[] for i in range(output_size[0] * output_size[1])]
    for i in range(0, y.shape[1]):
        a = tf.reduce_mean(tf.square(y[:, i] - predictions[:, i]))
        mse[i].append(a)

    loss = tf.reduce_mean(mse)
    global_step = tf.Variable(0)
    LR = tf.train.exponential_decay(0.1, global_step, int(m / batch_size), 0.96, staircase=True)
    train_op = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),optimizer=opt,learning_rate=LR)
    return predictions ,loss,train_op

def train(sess,x,y,):
    ds = tf.data.Dataset.from_tensor_slices((x,y))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    x,y = ds.make_one_shot_iterator().get_next()
    # x = x.eval(session = sess)
    # y = y.eval(session = sess)
    with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
        _, train_op, lossor = lstm_model(x, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(train_step):
        _, loss = sess.run([train_op,lossor])
        if i % 100 ==0:
            print('lost: ' + str(i), loss)
def test(sess, x, y, test_step):
    pred_y = np.zeros((test_step * batch_size,  y.shape[1]))
    errorcsv = np.zeros((test_step * batch_size, 3))
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size)
    x, y = ds.make_one_shot_iterator().get_next()
    # x = x.eval(session=sess)
    # y = y.eval(session=sess)
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        prediction, _, _ = lstm_model(x, [], False)#为了使得测试的时候lstm调用的参数值是训练好的，必须设置reuse=True
    for i in range(test_step):
        pred, ys = sess.run([prediction, y])#不能写y = 因为y和prediction被run以后就是ndarray,y这个变量已经存在了，类型是tensor，没法赋值，要能涵盖赋值，必须是同类型
        # for d in range(0, test_y.shape[1]):
        pred_y[i * batch_size:(i + 1) * batch_size, :] = pred

        mse = np.zeros((ys.shape[0],ys.shape[1]))
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
        MSE = np.mean(mse, axis = 1)
        MAE = np.mean(mae, axis = 1)
        MAPE = np.mean(mape, axis = 1)
        ERROR = [MSE, MAE, MAPE]
        # print("teststep:", i, ERROR)
        for e in range(0, 3):
            errorcsv[i*batch_size:(i+1)*batch_size, e] = ERROR[e]#这是list，没法赋给array直接

    return pred_y, errorcsv

if __name__=='__main__':

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    x = [[] for i in range(10000)]
    y = [[] for i in range(10000)]
    x2 = [[] for i in range(10000)]
    y2 = [[] for i in range(10000)]

    j = 0
    k = 0
    for i in range(0, len(sq_set)):
        if len(sq_set[i]) - time_step > 1:
            A, B, C, D = generate_data(sq_setscalar[i], sq_set[i])#这样算完，每一个x[i].shape is [batch_num[i], time_step, input_size]
            if A != []:
                x[j] = A
                y[j] = B
                j = j + 1
            if C!= []:
                x2[k] = C
                y2[k] = D
                k = k + 1

    xs = np.array(x[0])
    ys = np.array(y[0])
    x2s = np.array(x2[0])
    y2s = np.array(y2[0])


    for i in range(1, j):
        xs = np.vstack(((xs, np.array(x[i]))))
        ys = np.vstack(((ys, np.array(y[i]))))

    for i in range(1, k):
        x2s = np.vstack(((x2s, np.array(x2[i]))))
        y2s = np.vstack(((y2s, np.array(y2[i]))))


    train_end = int(len(xs) * 0.75)
    test_end = int(len(xs))
    train_step = 3000
    test_step = (test_end - train_end) // batch_size

    train_x = xs[0:train_end]
    train_y = ys[0:train_end]
    test_x = xs[train_end:test_end]
    test_y = ys[train_end:test_end]

    train_end2 = int(len(x2s) * 0.75)
    test_end2 = int(len(x2s))
    test_step2 = (test_end2 - train_end2) // batch_size

    train_x2 = x2s[0:train_end2]
    train_y2 = y2s[0:train_end2]
    test_x2 = x2s[train_end2:test_end2]
    test_y2 = y2s[train_end2:test_end2]



    with open(r"D:\轨迹预测\prediction\SVR\test_xo.pkl", 'wb') as f:
        pickle.dump(test_x, f)
    with open(r"D:\轨迹预测\prediction\SVR\test_yo.pkl", 'wb') as f:
        pickle.dump(test_y, f)

    with open(r"D:\轨迹预测\prediction\SVR\test_xu.pkl", 'wb') as f:
        pickle.dump(x2s, f)
    with open(r"D:\轨迹预测\prediction\SVR\test_yu.pkl", 'wb') as f:
        pickle.dump(y2s, f)

    Xo = np.reshape(test_x, [-1, 5])#周期10
    dt = pd.DataFrame(Xo)
    dt.to_csv(r"D:\轨迹预测\prediction\GM\test_xo.csv")
    dt = pd.DataFrame(test_y)
    dt.to_csv(r"D:\轨迹预测\prediction\GM\test_yo.csv")

    Xu = np.reshape(x2s, [-1, 5])#周期10
    dt = pd.DataFrame(Xu)
    dt.to_csv(r"D:\轨迹预测\prediction\GM\test_xu.csv")
    dt = pd.DataFrame(y2s)
    dt.to_csv(r"D:\轨迹预测\prediction\GM\test_yu.csv")

    root = r"D:\轨迹预测\prediction"
    sub = "GRU"
    # optimizer = ["SGD", "Adagrad", "Momentum"]
    optimizer = ["Adagrad"]

    MAEscalaro = np.zeros(3)
    MAEscalaru = np.zeros(3)

    for n in range(0, 3):
        for p in range(0, len(optimizer)):
            opt = optimizer[p]
            b = os.path.exists(root + "\\" + sub + "\\" + opt)
            if b:
                print("path exist")
            else:
                os.makedirs(root + "\\" + sub + "\\" + opt)

            m = len(xs)
            train_y1 = np.zeros((train_y.shape[0], 1), dtype=np.float32)
            test_y1 = np.zeros((test_y.shape[0], 1), dtype=np.float32)
            for i in range(train_y.shape[0]):
                train_y1[i, 0] = train_y[i, n]

            for i in range(test_y.shape[0]):
                test_y1[i, 0] = test_y[i, n]

            train(sess, train_x, train_y1)
            pred_yo, errorcsvo = test(sess, test_x, test_y1, test_step)
            MAEscalaro[n] = np.mean(errorcsvo[:, 1])

            m = len(x2s)
            train_yu = np.zeros((train_y2.shape[0], 1), dtype=np.float32)
            test_yu = np.zeros((test_y2.shape[0], 1), dtype=np.float32)
            for i in range(train_y2.shape[0]):
                train_yu[i, 0] = train_y2[i, n]

            for i in range(test_y2.shape[0]):
                test_yu[i, 0] = test_y2[i, n]
            train(sess, train_x2, train_yu)
            pred_yu, errorcsvu = test(sess, test_x2, test_yu, test_step2)

            MAEscalaru[n] = np.mean(errorcsvu[:, 1])

            # dt = pd.DataFrame(errorcsvo)
            # dt.to_csv(root + "\\" + sub + "\\" + "error_o.csv")
            # dt = pd.DataFrame(errorcsvu)
            # dt.to_csv(root + "\\" + sub + "\\" + "error_u.csv")
            # dt = pd.DataFrame(pred_yo)
            # dt.to_csv(root + "\\" + sub + "\\" + "pred_yo.csv")
            # dt = pd.DataFrame(pred_yu)
            # dt.to_csv(root + "\\" + sub + "\\" + "pred_yu.csv")

    dt = pd.DataFrame(MAEscalaro)
    dt.to_csv(root + "\\" + sub + "\\" + "MAEo.csv")
    dt = pd.DataFrame(MAEscalaru)
    dt.to_csv(root + "\\" + sub + "\\" + "MAEu.csv")