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
# or_data = np.zeros((len(flightTime), 5))
# for i in range(0, or_data.shape[0]):
#     or_data[i, 0] = altitude[i]
#     or_data[i, 1] = speed[i]
#     or_data[i, 2] = xz[i]
#     or_data[i, 3] = yz[i]
#     or_data[i, 4] = vy[i]
#     # or_data[i, 5] = v[i]


# scalar = MinMaxScaler(feature_range=(0, 1))
# or_data = scalar.fit_transform(or_data).squeeze()

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
diff = np.zeros((len(sq_set), 3))
lower = np.zeros((len(sq_set), 3))
upper = np.zeros((len(sq_set), 3))
ydian = np.zeros((1, or_data.shape[1]))
for i in range(0, len(sq_set)):
    s = np.array(sq_set_diff[i])
    ydian = np.vstack((ydian, s))
    for j in range(0, 3):
        diff[i, j] = np.max(np.abs(s[:, j]))
        lower[i, j] = np.min(s[:, j])
        upper[i, j] = np.max(s[:, j])
with open(r"D:\轨迹预测\ydian.pkl", 'wb') as f:
    pickle.dump(ydian, f)

dt2 = pd.DataFrame(diff)
dt2.to_csv(r"D:\轨迹预测\prediction\diff.csv")
dt2 = pd.DataFrame(lower)
dt2.to_csv(r"D:\轨迹预测\prediction\lower.csv")
dt2 = pd.DataFrame(upper)
dt2.to_csv(r"D:\轨迹预测\prediction\upper.csv")


for i in range(0, len(sq_set)):
    scalar = MinMaxScaler(feature_range=(0, 1))
    sq_set_diff[i] = scalar.fit_transform(sq_set_diff[i]).squeeze()

time_step = 40
batch_size = 40
hidden_size = 15
layer_num = 3



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
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(hidden_size)
                                        for _ in range(layer_num)])
    outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    output = outputs[:, -1, :] #[batchsize,hiddensize]

    nonpredictions = tf.contrib.layers.fully_connected(output, output_size[0] * output_size[1], activation_fn = None )
    predictions = tf.nn.leaky_relu(nonpredictions,alpha = 0.2, name = None)
    if not is_training:
        return predictions ,None,None

    # predictions = tf.nn.dropout(predictions, 0.99)
    mse = [[] for i in range(output_size[0] * output_size[1])]
    for i in range(0, y.shape[1]):
        a = tf.reduce_mean(tf.square(y[:, i] - predictions[:, i]))
        mse[i].append(a)

    loss = tf.reduce_mean(mse)
    global_step = tf.Variable(0)
    LR = tf.train.exponential_decay(0.1, global_step, int(m / batch_size), 0.96, staircase=True)
    train_op = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),optimizer=opt,learning_rate=LR)
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
        # fig = plt.figure()
        # ax1 = plt.axes(projection='3d')
        # x1 = pred[:,1]
        # y1 = pred[:,2]
        # z1 = pred[:,0]
        # x2 = ys[:,1]
        # y2 = ys[:,2]
        # z2 = ys[:,0]
        # ax1.scatter3D(x1, y1, z1, cmap='red')  # 绘制散点图
        # ax1.plot3D(x1, y1, z1, 'black')  # 绘制空间曲线
        # ax1.scatter3D(x2, y2, z2, cmap='blue')
        # ax1.plot3D(x2, y2, z2, 'gray')
        # plt.show()

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

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

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


    # with open(r"D:\轨迹预测\train_x.pkl", 'wb') as f:
    #     pickle.dump(train_x, f)
    # with open(r"D:\轨迹预测\train_y.pkl", 'wb') as f:
    #     pickle.dump(train_y, f)
    # with open(r"D:\轨迹预测\test_x.pkl", 'wb') as f:
    #     pickle.dump(test_x, f)
    # with open(r"D:\轨迹预测\test_y.pkl", 'wb') as f:
    #     pickle.dump(test_y, f)


    # ceshiji = np.reshape(test_x, [-1, 5])
    # dt1 = pd.DataFrame(ceshiji)
    # dt1.to_csv(r'C:\Users\tangrong\Desktop\测试集x.csv')
    # dt2 = pd.DataFrame(test_y)
    # dt2.to_csv(r'C:\Users\tangrong\Desktop\测试集y.csv')
    root = r"D:\轨迹预测\prediction"
    sub = "GRU"
    # optimizer = ["SGD", "Adagrad", "Momentum"]
    optimizer = ["Adagrad"]

    MAEscalar = np.zeros((5, 3))
    output_size = ((1, 1))

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

            train(sess, train_x, train_y1)
            pred1[n], errorcsv1[n] = test(sess, test_x, test_y1, test_step)

            MAEscalar[n, 0] = np.mean(errorcsv1[n][:, 1])

        # with open(r"D:\轨迹预测\prediction\LSTM\pred1.pkl", 'wb') as f:
        #     pickle.dump(pred1, f)
        # with open(r"D:\轨迹预测\prediction\LSTM\errorcsv1.pkl", 'wb') as f:
        #     pickle.dump(errorcsv1, f)

        # with open(r"D:\轨迹预测\prediction\LSTM\pred1.pkl", 'rb') as f:
        #     pred1 = pickle.load(f)
        # with open(r"D:\轨迹预测\prediction\LSTM\errorcsv1.pkl", 'rb') as f:
        #     errorcsv1 = pickle.load(f)

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

            train(sess, train_x2, train_y21)
            pred2[n], errorcsv2[n] = test(sess, test_x2, test_y21, test_step2)

            MAEscalar[n, 1] = np.mean(errorcsv2[n][:, 1])

        # with open(r"D:\轨迹预测\prediction\LSTM\pred2.pkl", 'wb') as f:
        #     pickle.dump(pred2, f)
        # with open(r"D:\轨迹预测\prediction\LSTM\errorcsv2.pkl", 'wb') as f:
        #     pickle.dump(errorcsv2, f)
        #
        # with open(r"D:\轨迹预测\prediction\LSTM\pred2.pkl", 'rb') as f:
        #     pred2 = pickle.load(f)
        # with open(r"D:\轨迹预测\prediction\LSTM\errorcsv2.pkl", 'rb') as f:
        #     errorcsv2 = pickle.load(f)

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

            train(sess, train_x3, train_y31)
            pred3[n], errorcsv3[n] = test(sess, test_x3, test_y31, test_step3)
            MAEscalar[n, 2] = np.mean(errorcsv3[n][:, 1])

        # with open(r"D:\轨迹预测\prediction\LSTM\pred3.pkl", 'wb') as f:
        #     pickle.dump(pred3, f)
        # with open(r"D:\轨迹预测\prediction\LSTM\errorcsv3.pkl", 'wb') as f:
        #     pickle.dump(errorcsv3, f)
        #
        # # with open(r"D:\轨迹预测\prediction\LSTM\pred3.pkl", 'rb') as f:
        # #     pred3 = pickle.load(f)
        # # with open(r"D:\轨迹预测\prediction\LSTM\errorcsv3.pkl", 'rb') as f:
        # #     errorcsv3 = pickle.load(f)


    #
    #
    #     # dt = pd.DataFrame(errorcsv1)
    #     # dt.to_csv(root + "\\" + sub + "\\" + "error_1.csv")
    #     # dt = pd.DataFrame(errorcsv2)
    #     # dt.to_csv(root + "\\" + sub + "\\" + "error_2.csv")
    #     # dt = pd.DataFrame(errorcsv3)
    #     # dt.to_csv(root + "\\" + sub + "\\" + "error_3.csv")
    #     #
    #     #
    #     # dt = pd.DataFrame(pred_y1)
    #     # dt.to_csv(root + "\\" + sub + "\\" + "pred_y1.csv")
    #     # dt = pd.DataFrame(pred_y2)
    #     # dt.to_csv(root + "\\" + sub + "\\" + "pred_y2.csv")
    #     # dt = pd.DataFrame(pred_y3)
    #     # dt.to_csv(root + "\\" + sub + "\\" + "pred_y3.csv")

    # for n in range(0, 5):
    #     MAEscalar[n, 0] = np.mean(errorcsv1[n][:, 1])
    #     MAEscalar[n, 1] = np.mean(errorcsv2[n][:, 1])
    #     MAEscalar[n, 2] = np.mean(errorcsv3[n][:, 1])
    dt = pd.DataFrame(MAEscalar)
    dt.to_csv(root + "\\" + sub + "\\" + "MAE123.csv")







