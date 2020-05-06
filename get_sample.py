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
import xlsxwriter


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
    scalar = MinMaxScaler(feature_range=(-1, 1))
    sq_set[i] = scalar.fit_transform(sq_set[i]).squeeze()



batch_size = 40



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

if __name__=='__main__':

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    MAEscalar = np.zeros((6, 70))
    for t in range(5, 55, 5):
        time_step = t
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

        root = r"D:\轨迹预测\prediction\SVR\scalar"
        with open(root + '\\' + 'train_x' + str(time_step) + ".pkl", 'wb') as f:
            pickle.dump(train_x, f)
        with open(root + '\\' + 'train_y' + str(time_step) + ".pkl", 'wb') as f:
            pickle.dump(train_y, f)
        with open(root + '\\' + 'test_x' + str(time_step) + ".pkl", 'wb') as f:
            pickle.dump(test_x, f)
        with open(root + '\\' + 'test_y' + str(time_step) + ".pkl",'wb') as f:
            pickle.dump(test_y, f)