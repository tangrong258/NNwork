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
for i in range(0, len(sq_set)):
    for j in range(1, len(sq_set[i])):
        sq_set[i][j][0] = sq_set[i][j][0] - sq_set[i][0][0]
        sq_set[i][j][1] = sq_set[i][j][1] - sq_set[i][0][1]
        sq_set[i][j][2] = sq_set[i][j][2] - sq_set[i][0][2]

for i in range(0, len(sq_set)):
    sq_set[i][0][0] = 0
    sq_set[i][0][1] = 0
    sq_set[i][0][2] = 0



diff = np.zeros((len(sq_set), 3))
lower = np.zeros((len(sq_set), 3))
upper = np.zeros((len(sq_set), 3))
ydian = np.zeros((1, or_data.shape[1]))
for i in range(0, len(sq_set)):
    s = np.array(sq_set[i])
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
    sq_set[i] = scalar.fit_transform(sq_set[i]).squeeze()


time_step = 10
batch_size = 50
hidden_size = 20
layer_num = 2
output_size = ((1,3))


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
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
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
    with tf.variable_scope("model",reuse = tf.AUTO_REUSE):
        _, train_op, lossor = lstm_model(x, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(train_step):
        _, loss = sess.run([train_op,lossor])
        if i % 100 ==0:
            print('lost: ' + str(i), loss)
def test(sess,x,y):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size)
    x, y = ds.make_one_shot_iterator().get_next()
    # x = x.eval(session=sess)
    # y = y.eval(session=sess)
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        prediction,_,_ = lstm_model(x, [], False)#为了使得测试的时候lstm调用的参数值是训练好的，必须设置reuse=True
    for i in range(test_step):
        pred, ys = sess.run([prediction, y])#不能写y = 因为y和prediction被run以后就是ndarray,y这个变量已经存在了，类型是tensor，没法赋值，要能涵盖赋值，必须是同类型
        # for d in range(0, test_y.shape[1]):
        pred_y[i * batch_size:(i + 1) * batch_size, :] = pred
        fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        x1 = pred[:,1]
        y1 = pred[:,2]
        z1 = pred[:,0]
        x2 = ys[:,1]
        y2 = ys[:,2]
        z2 = ys[:,0]
        ax1.scatter3D(x1, y1, z1, cmap='red')  # 绘制散点图
        ax1.plot3D(x1, y1, z1, 'black')  # 绘制空间曲线
        ax1.scatter3D(x2, y2, z2, cmap='blue')
        ax1.plot3D(x2, y2, z2, 'gray')
        plt.show()

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
        MSE = np.mean(mse,axis = 1)
        MAE = np.mean(mae,axis = 1)
        MAPE = np.mean(mape,axis = 1)
        ERROR = [MSE,MAE,MAPE]
        print("teststep:",i,ERROR)
        for e in range(0, 3):
            errorcsv[i*batch_size:(i+1)*batch_size, e + 3*p] = ERROR[e]

if __name__=='__main__':

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    x = [[] for i in range(10000)]
    y = [[] for i in range(10000)]
    j = 0
    for i in range(0, len(sq_set)):
        if len(sq_set[i]) - time_step > 1:
            x[j], y[j] = generate_data(sq_set[i])#这样算完，每一个x[i].shape is [batch_num[i], time_step, input_size]
            j = j + 1

    xs = np.array(x[0])
    ys = np.array(y[0])

    for i in range(1, j):
        xs = np.vstack((xs, np.array(x[i])))
        ys = np.vstack((ys, np.array(y[i])))

    m = len(xs)

    train_end = int(len(xs) * 0.75)
    test_end = int(len(xs))
    train_step = 3000
    test_step = (test_end - train_end) // batch_size

    train_x = xs[0:train_end]
    train_y = ys[0:train_end]
    test_x = xs[train_end:test_end]
    test_y = ys[train_end:test_end]

    with open(r"D:\轨迹预测\train_x.pkl", 'wb') as f:
        pickle.dump(train_x, f)
    with open(r"D:\轨迹预测\train_y.pkl", 'wb') as f:
        pickle.dump(train_y, f)
    with open(r"D:\轨迹预测\test_x.pkl", 'wb') as f:
        pickle.dump(test_x, f)
    with open(r"D:\轨迹预测\test_y.pkl", 'wb') as f:
        pickle.dump(test_y, f)


    ceshiji = np.reshape(test_x, [-1, 5])
    dt1 = pd.DataFrame(ceshiji)
    dt1.to_csv(r'C:\Users\tangrong\Desktop\测试集x.csv')
    dt2 = pd.DataFrame(test_y)
    dt2.to_csv(r'C:\Users\tangrong\Desktop\测试集y.csv')
    root = r"D:\轨迹预测\prediction"
    sub = "LSTM"
    # optimizer = ["SGD", "Adagrad", "Momentum"]
    optimizer = ["Adagrad"]
    pred_y = np.zeros((test_step * batch_size, test_y.shape[1]))
    errorcsv = np.zeros((test_step * batch_size, len(optimizer) * 3))
    for p in range(0, len(optimizer)):
        opt = optimizer[p]
        b = os.path.exists(root + "\\" + sub + "\\" + opt)
        if b:
            print("path exist")
        else:
            os.makedirs(root + "\\" + sub + "\\" + opt)


        train(sess, train_x, train_y)
        test(sess, test_x, test_y)

    # dt = pd.DataFrame(errorcsv, columns=[optimizer[0],'','',optimizer[1],'','',optimizer[2], '', ''])
    # dt.to_csv(root + "\\" + sub + "\\" + "error.csv")
    # dt1 = pd.DataFrame(pred_y, columns=[optimizer[0], '', '', optimizer[1], '', '', optimizer[2], '', ''])
    # dt1.to_csv(root + "\\" + sub + "\\" + "pred_y.csv")
    dt = pd.DataFrame(errorcsv)
    dt.to_csv(root + "\\" + sub + "\\" + "error.csv")
    dt1 = pd.DataFrame(pred_y)
    dt1.to_csv(root + "\\" + sub + "\\" + "pred_y.csv")

