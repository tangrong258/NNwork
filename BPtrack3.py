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



for i in range(0, len(sq_set)):
    scalar = MinMaxScaler(feature_range=(0, 1))
    sq_set_diff[i] = scalar.fit_transform(sq_set_diff[i]).squeeze()

batch_size = 256
output_size = 1



def generate_data1(sq):
    x = []
    y = []
    y2 = []
    y3 = []
    sq = np.array(sq)
    for i in range(0, len(sq)-time_step-2):
        x.append([sq[i:i + time_step]])
        y.append([sq[i + time_step, :]])
        y2.append([sq[i + time_step + 1, :]])
        y3.append([sq[i + time_step + 2, :]])
    xs = np.array(x, dtype=np.float32).squeeze()
    ys = np.array(y, dtype=np.float32).squeeze()
    y2s = np.array(y2, dtype=np.float32).squeeze()
    y3s = np.array(y3, dtype=np.float32).squeeze()
    return xs, ys, y2s, y3s



def lstm_model(x, y,  is_training):
    input2hidden = tf.nn.sigmoid(tf.matmul(x, w1) + b1)  # sigmoid激活函数
    h1_h2 = tf.nn.sigmoid(tf.matmul(input2hidden, w2) + b2)
    h2_h3 = tf.nn.sigmoid(tf.matmul(h1_h2, w3) + b3)
    nonpredictions = tf.nn.sigmoid(tf.matmul(h2_h3, w4) + b4)
    predictions = tf.nn.leaky_relu(nonpredictions, alpha=0.1, name=None)
    if not is_training:
        return predictions, None, None

    predictions = tf.nn.dropout(predictions, 0.90)

    if np.size(output_size) != 1:
        mse = [[] for _ in range(output_size[0] * output_size[1])]
    else:
        mse = [[] for _ in range(output_size)]

    for i in range(0, y.shape[1]):
        a = tf.reduce_mean(tf.square(y[:, i] - predictions[:, i]))
        mse[i].append(a)

    loss = tf.reduce_mean(mse)
    global_step = tf.Variable(0)
    LR = tf.train.exponential_decay(0.1, global_step, int(m / batch_size), 0.96, staircase=True)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer=opt, learning_rate=LR)
    return predictions, loss, train_op

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
            errorcsv[i*batch_size:(i+1)*batch_size, e] = ERROR[e]  # 这是list，没法赋给array直接

    return pred_y, errorcsv


if __name__ == '__main__':

    tf.reset_default_graph()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    time_step = 10
    input_size = time_step * 5  # 输入节点个数

    hidden_size = 2 * input_size + 1  # 隐层个数,采用经验公式2d+1
    # 初始化权值和阈值
    w1 = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=1, seed=1))  # seed设定随机种子，保证每次初始化相同数据
    b1 = tf.Variable(tf.constant(0.0, shape=[hidden_size]))

    w2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=1, seed=1))
    b2 = tf.Variable(tf.constant(0.0, shape=[hidden_size]))

    w3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=1, seed=1))
    b3 = tf.Variable(tf.constant(0.0, shape=[hidden_size]))

    w4 = tf.Variable(tf.random_normal([hidden_size, output_size], stddev=1, seed=1))
    b4 = tf.Variable(tf.constant(0.0, shape=[output_size]))

    x = [[] for i in range(10000)]
    y = [[] for i in range(10000)]
    y2 = [[] for i in range(10000)]
    y3 = [[] for i in range(10000)]


    j = 0
    for i in range(0, len(sq_set)):
        if len(sq_set[i]) - time_step - 2 > 1: #caution
            x[j], y[j], y2[j], y3[j] = generate_data1(sq_set_diff[i])#这样算完，每一个x[i].shape is [batch_num[i], time_step, input_size]
            j = j + 1

    xs = np.array(x[0])
    ys = np.array(y[0])
    y2s = np.array(y2[0])
    y3s = np.array(y3[0])

    for i in range(1, j):
        xs = np.vstack(((xs, np.array(x[i]))))
        ys = np.vstack(((ys, np.array(y[i]))))
        y2s = np.vstack(((y2s, np.array(y2[i]))))
        y3s = np.vstack(((y3s, np.array(y3[i]))))
    m = len(xs)

    train_end = int(len(xs) * 0.5)
    test_end = int(len(xs))
    train_step = 4000
    test_step = (test_end - train_end) // batch_size

    train_x = xs[0:train_end]
    train_y = ys[0:train_end]
    test_x = xs[train_end:test_end]
    test_y = ys[train_end:test_end]

    root = r"D:\轨迹预测\prediction"
    sub = "BP"
    optimizer = ["Adagrad"]

    MAEscalar = np.zeros((5, 3))

    for p in range(0, len(optimizer)):
        opt = optimizer[p]
        b = os.path.exists(root + "\\" + sub + "\\" + opt)
        if b:
            print("path exist")
        else:
            os.makedirs(root + "\\" + sub + "\\" + opt)

        pred1 = [[] for i in range(test_y.shape[1])]
        errorcsv1 = [[] for i in range(test_y.shape[1])]

        for n in range(0, test_y.shape[1]):
            train_y1 = np.zeros((train_y.shape[0], 1), dtype=np.float32)
            test_y1 = np.zeros((test_y.shape[0], 1), dtype=np.float32)
            for i in range(train_y.shape[0]):
                train_y1[i, 0] = train_y[i, n]

            for i in range(test_y.shape[0]):
                test_y1[i, 0] = test_y[i, n]

            train_xv = np.reshape(train_x, [train_x.shape[0], -1])
            train(sess, train_xv, train_y1)

            test_xv = np.reshape(test_x, [test_x.shape[0], -1])
            pred1[n], errorcsv1[n] = test(sess, test_xv, test_y1, test_step)

            MAEscalar[n, 0] = np.mean(errorcsv1[n][:, 1])


        pred_y1 = np.array(pred1).squeeze().T
        y2_org = y2s[train_end:train_end+len(pred_y1)]
        x2 = np.delete(test_x, 0, 1)
        x2_org = np.zeros(((pred_y1.shape[0], test_x.shape[1], test_x.shape[2])),
                          dtype=np.float32)  # python默认float64，LSTM需要32
        for i in range(0, len(pred_y1)):
            x2_org[i] = np.vstack(((x2[i], pred_y1[i])))

        train_end2 = int(len(x2_org) * 0.5)
        test_end2 = int(len(x2_org))
        test_step2 = (test_end2 - train_end2) // batch_size

        train_x2 = x2_org[0:train_end2]
        train_y2 = y2_org[0:train_end2]
        test_x2 = x2_org[train_end2:test_end2]
        test_y2 = y2_org[train_end2:test_end2]

        pred2 = [[] for i in range(test_y2.shape[1])]
        errorcsv2 = [[] for i in range(test_y2.shape[1])]


        for n in range(0, test_y2.shape[1]):
            train_y21 = np.zeros((train_y2.shape[0], 1), dtype=np.float32)
            test_y21 = np.zeros((test_y2.shape[0], 1), dtype=np.float32)
            for i in range(train_y2.shape[0]):
                train_y21[i, 0] = train_y2[i, n]

            for i in range(test_y2.shape[0]):
                test_y21[i, 0] = test_y2[i, n]

            train_x2v = np.reshape(train_x2, [train_x2.shape[0], -1])
            train(sess, train_x2v, train_y21)

            test_x2v = np.reshape(test_x2, [test_x2.shape[0], -1])
            pred2[n], errorcsv2[n] = test(sess, test_x2v, test_y21, test_step2)

            MAEscalar[n, 1] = np.mean(errorcsv2[n][:, 1])


        pred_y2 = np.array(pred2).squeeze().T
        y3_org = y3s[train_end2:train_end2 + len(pred_y2)]
        x3 = np.delete(test_x2, 0, 1)
        x3_org = np.zeros(((pred_y2.shape[0], test_x2.shape[1], test_x2.shape[2])),
                          dtype=np.float32)  # python默认float64， LSTM需要32
        for i in range(0, len(pred_y2)):
            x3_org[i] = np.vstack(((x3[i], pred_y2[i])))

        train_end3 = int(len(x3_org) * 0.5)
        test_end3 = int(len(x3_org))
        test_step3 = (test_end3 - train_end3) // batch_size

        train_x3 = x3_org[0:train_end3]
        train_y3 = y3_org[0:train_end3]
        test_x3 = x3_org[train_end3:test_end3]
        test_y3 = y3_org[train_end3:test_end3]

        pred3 = [[] for i in range(test_y3.shape[1])]
        errorcsv3 = [[] for i in range(test_y3.shape[1])]


        for n in range(0, test_y3.shape[1]):
            train_y31 = np.zeros((train_y3.shape[0], 1), dtype=np.float32)
            test_y31 = np.zeros((test_y3.shape[0], 1), dtype=np.float32)
            for i in range(train_y3.shape[0]):
                train_y31[i, 0] = train_y3[i, n]

            for i in range(test_y3.shape[0]):
                test_y31[i, 0] = test_y3[i, n]

            train_x3v = np.reshape(train_x3, [train_x3.shape[0], -1])
            train(sess, train_x3v, train_y31)

            test_x3 = np.reshape(test_x3, [test_x3.shape[0], -1])
            pred3[n], errorcsv3[n] = test(sess, test_x3, test_y31, test_step3)
            MAEscalar[n, 2] = np.mean(errorcsv3[n][:, 1])

    dt = pd.DataFrame(MAEscalar)
    dt.to_csv(root + "\\" + sub + "\\" + "MAE123.csv")







