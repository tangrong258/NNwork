import numpy as np
import csv
import math
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, MultipleLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from matplotlib.font_manager import FontProperties as FP
plt.rc('font', family='Times New Roman')
font_TNM = FP(fname=r"C:\Windows\Fonts\times.ttf", size=10.5)

Altitude_scalar = 12.44
Latitude_scalar = 52.13
Longitude_scalar = 39.78
Altitude_xishu = 1.32
Latitude_xishu = 1.08
Longitude_xishu = 1.08


# pred_true_values line
"""

data_LSTM = pd.read_csv(r"D:\轨迹预测\prediction\LSTM\pred_y.csv")
data_GRU = pd.read_csv(r"D:\轨迹预测\prediction\GRU\pred_y.csv")
data_BP = pd.read_csv(r"D:\轨迹预测\prediction\BP\pred_y10.csv")
data_SVR = pd.read_csv(r"D:\轨迹预测\prediction\SVR\pred_y.csv")
data_GM = pd.read_csv(r"D:\轨迹预测\prediction\GM\pred_y.csv")
data_ARIMA = pd.read_csv(r"D:\轨迹预测\prediction\ARIMA\pred_y.csv")
data_testy = pd.read_csv(r"D:\轨迹预测\prediction\ARIMA\test_y500.csv")

pred_LSTM = data_LSTM._values[:, 1:4]
ys = data_LSTM._values[:, 4:7]
pred_GRU = data_GRU._values[:, 1:4]
pred_BP = data_BP._values[:, 1:4]
pred_SVR = data_SVR._values[:, 1:4]
pred_GM = data_GM._values[:, 1:4]
pred_ARIMA = data_ARIMA._values[:, 1:4]
ysarima = data_testy._values[:, 1:4]

# scalar = MinMaxScaler(feature_range=(0, 1))
# pred_ARIMA = scalar.fit_transform(pred_ARIMA).squeeze()

def plottime_value(ys, pred, object, max_value, i, model_name, color_line):
    Z_start = 1000
    Z_end = 1300
    fig = plt.figure(figsize=(2.96, 2.22))
    if np.max(pred) > 10:
        plt.plot(np.arange(Z_start, Z_end), ysarima[0:Z_end - Z_start, i], 'grey', linestyle='-', linewidth='1.5')
    else:
        plt.plot(np.arange(Z_start, Z_end), ys[Z_start:Z_end, i], 'grey', linestyle='-', linewidth='1.5')

    if len(pred) > 1000:
        plt.plot(np.arange(Z_start + 50, Z_end), pred[Z_start + 50:Z_end, i], color_line, linestyle='--', linewidth='1.25')
    # plt.plot(np.arange(Z_start + 50, Z_end), pred2[Z_start + 50:Z_end, i], 'lime', linestyle='--', linewidth='1.25')
    # plt.plot(np.arange(Z_start + 50, Z_end), pred3[50: Z_end-Z_start, i], 'teal', linestyle='--', linewidth='1.25')
    # plt.plot(np.arange(Z_start + 50, Z_end), pred4[Z_start + 50:Z_end, i], 'violet', linestyle='--', linewidth='1.25')
    else:
        plt.plot(np.arange(Z_start + 50, Z_end), pred[50: Z_end - Z_start, i], color_line, linestyle='--', linewidth='1.25')

    plt.xticks(np.arange(Z_start, Z_end+50, 50), np.arange(0, int((Z_end-Z_start+50)*5), 250), fontsize=10.5)

    if np.max(pred) < 10:
        plt.yticks(np.arange(0, 1.2, 0.2), np.arange(0, max_value+int(max_value/5), int(max_value/5)), fontsize=10.5)
    else:
        lower = np.min(ysarima[:, i])
        upper = np.max(ysarima[:, i])
        plt.yticks(np.arange(lower,  upper + int((upper - lower) / 5), int((upper - lower) / 5)),
                   np.arange(0, max_value + int(max_value / 5), int(max_value / 5)), fontsize=10.5)


    plt.xlabel(u'time', fontsize=10.5)
    plt.ylabel(object, fontsize=10.5)
    plt.grid(False)
    plt.savefig(r'D:\轨迹预测\prediction'+'\\' + model_name + '\\' + object + '.png', dpi=300)
    plt.show()

object = [u'Altitude', u'Latitude', u'Longitude']
max_value = np.array([75, 165, 150])
model_name = ['LSTM', 'GRU', 'BP', 'SVR', 'GM', 'ARIMA']
Color_line = ['red', 'violet', 'blue', 'lime', 'cyan', 'orange']

for i in range(0, 3):
    for k in range(0, len(model_name)):
        plottime_value(ys, eval('pred_' + model_name[k]), object[i], max_value[i], i, model_name[k], Color_line[k])

"""


# pred_true_3D
"""
D3_start = 0
D3_end = 100
fig2 = plt.figure(figsize=(3.5, 2.5))
ax1 = plt.axes(projection='3d')
# ax1.scatter3D(pred[D3_start:D3_end, 2], pred[D3_start:D3_end, 1], pred[D3_start:D3_end, 0], 'orange', s=6.0)
ax1.plot3D(pred[D3_start:D3_end, 2], pred[D3_start:D3_end, 1], pred[D3_start:D3_end, 0], 'black', linestyle='--', linewidth='1.5', label='predcition')
# ax1.scatter3D(ys[D3_start:D3_end, 2], ys[D3_start:D3_end, 1], ys[D3_start:D3_end, 0], 'blue', s=6.0)
ax1.plot3D(ys[D3_start:D3_end, 2], ys[D3_start:D3_end, 1], ys[D3_start:D3_end, 0], 'grey', linestyle='-', linewidth='1.5', label='real_value')
ax1.set_xticks(np.linspace(np.min(pred[D3_start:D3_end, 2]), np.max(pred[D3_start:D3_end, 2]), 5))
ax1.set_xticklabels(np.arange(0, 10, 2), fontsize=10.5)
ax1.set_yticks(np.linspace(np.min(pred[D3_start:D3_end, 1]), np.max(pred[D3_start:D3_end, 1]), 5))
ax1.set_yticklabels(np.arange(0, 10, 2), fontsize=10.5)
ax1.set_zticks(np.linspace(np.min(pred[D3_start:D3_end, 0]), np.max(pred[D3_start:D3_end, 0]), 5))
ax1.set_zticklabels(np.arange(0, 10, 2), fontsize=10.5)
ax1.set_xlabel(u'Longitude', fontsize=10.5)
ax1.set_ylabel(u'Latitude', fontsize=10.5)
ax1.set_zlabel(u'Altitude', fontsize=10.5)
ax1.grid(True)
# plt.savefig(root + "\\" + '3D' + ".png", dpi=300)
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.show()
"""



# hidden_layer1_8
"""

data_laynum = pd.read_excel(r"C:\\Users\\TR\Desktop\轨迹预测实验结果.xlsx", sheet_name='hlayer1_8')
data_Altitude = data_laynum._values[0:20:5, 1:]*Altitude_scalar*Altitude_xishu
data_Latitude = data_laynum._values[1:21:5, 1:]*Latitude_scalar*Latitude_xishu
data_Longitude = data_laynum._values[2:22:5, 1:]*Longitude_scalar*Longitude_xishu
data_time = data_laynum._values[3:23:5, 1:]

def plot_hidden_layer_opt(Altitude, Latitude, Longitude, title, time, time_lower, time_upper, time_unit):
    fig = plt.figure(figsize=(2.96, 2.22))
    ax1 = fig.add_subplot(111)
    # ax1.scatter(np.arange(1, 9), Altitude, marker='^', color='lime', s=10.0)
    ax1.plot(np.arange(1, 9), Altitude, color='lime', linestyle='--', linewidth=1.25, marker='^', markersize=6.0)
    # ax1.scatter(np.arange(1, 9), Latitude, marker='s', color='cyan', s=10.0)
    ax1.plot(np.arange(1, 9), Latitude, color='cyan', linestyle='--', linewidth=1.25, marker='s', markersize=6.0)
    # ax1.scatter(np.arange(1, 9), Longitude, marker='d', color='orange', s=10.0)
    ax1.plot(np.arange(1, 9), Longitude, color='orange', linestyle='--', linewidth=1.25, marker='d', markersize=6.0)
    ax1.set_xticks(np.arange(1, 9))
    ax1.set_xticklabels(np.arange(1, 9), fontsize=10.5)
    if np.max(Latitude) > 5.8 or np.max(Longitude) > 5.8:
        upper = 8
    else:
        upper = 6
    ax1.set_yticks(np.arange(0, upper+2, 2))
    ax1.set_yticklabels(np.arange(0, upper+2, 2), fontsize=10.5)

    ax1.set_xlabel(u'Number of hidden layers', fontsize=10.5)
    ax1.set_ylabel(u'MAE (m)', fontsize=10.5)
    # ax1.set_title(title)
    ax1.grid(False)
    ax2 = ax1.twinx()
    # ax2.scatter(np.arange(1, 9), time, marker='o', color='red', s=10.0)
    ax2.plot(np.arange(1, 9), time, color='grey', linestyle='-', linewidth=1.25, marker='o', markersize=6.0)
    ax2.set_yticks(np.arange(time_lower, time_upper, time_unit))
    ax2.set_yticklabels(np.arange(time_lower, time_upper, time_unit), fontsize=10.5)
    ax2.set_ylabel(u'Training time (sec)', fontsize=10.5)
    plt.tight_layout()
    # plt.subplots_adjust(0.02, 0.02, 0.03, 0.03, 2.22, 2.96)
    plt.savefig(r'D:\轨迹预测\prediction\GRU' +'\\'+ 'hlayer_number' + title + '.png', dpi=300)
    plt.show()

title_set = ['Step5&Cell5', 'Step5&Cell30', 'Step40&Cell5', 'Step40&Cell30']
for i in range(len(data_time)):
    time_lower = math.floor(np.min(data_time[i])/10)*10
    time_upper = math.ceil(np.max(data_time[i])/10)*10
    if time_upper <= 90:
        y2int_num = 4
    if time_upper >90 and time_upper<=130:
        y2int_num = 3
    if time_upper >130:
        y2int_num = 6
    time_unit = math.ceil((time_upper - time_lower)/y2int_num/10)*10
    time_upper_pro = time_lower + time_unit*(y2int_num+1)
    plot_hidden_layer_opt(data_Altitude[i], data_Latitude[i], data_Longitude[i], title_set[i],
                          data_time[i], time_lower, time_upper_pro, time_unit)

"""



# hidden_size & time_step 3D views
"""

data_csize_tstep = pd.read_excel(r'C:\\Users\TR\Desktop\轨迹预测实验结果.xlsx', sheet_name='cell_step_2000')
data_Altitude = data_csize_tstep._values[0::4, 1:]*Altitude_scalar*Altitude_xishu
data_Latitude = data_csize_tstep._values[1::4, 1:]*Latitude_scalar*Latitude_xishu
data_Longitude = data_csize_tstep._values[2::4, 1:]*Longitude_scalar*Longitude_xishu
data_Time = data_csize_tstep._values[3::4, 1:]
data_plot = [data_Altitude, data_Latitude, data_Longitude, data_Time]
object = ['MAE (m)', 'MAE (m)', 'MAE (m)', 'Training time (sec)']
cell_num = np.arange(5, 35, 5)
time_step = np.hstack((np.arange(5, 10, 1), np.arange(10, 24, 2), np.arange(25, 45, 5)))
x_lower = time_step[0]
x_upper = time_step[-1]
y_lower = cell_num[0]
y_upper = cell_num[-1]


def pltsurf(X, Y, Z, xticks, yticks, zmajorLocator, zmajorFormatter, object):
    fig = plt.figure(figsize=(3.0, 3.4))
    ax1 = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('cool'), antialiased=True)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=10.5)
    ax1.set_xlabel(u'Time step', fontsize=10.5)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks, fontsize=10.5)
    ax1.set_ylabel(u'Hidden size', fontsize=10.5)
    ax1.zaxis.set_major_locator(zmajorLocator)
    ax1.zaxis.set_major_formatter(zmajorFormatter)
    ax1.set_zlabel(object, fontsize=10.5)
    position = fig.add_axes([0.05, 0.25, 0.06, 0.45])
    cb = fig.colorbar(surf, shrink=0.5, aspect=10, cax=position, orientation='vertical')#高度和宽度
    cb.ax.tick_params(labelsize=10.5)
    cb_ticks = ax1.get_zticks()
    # cb_ticks_1f = cb_ticks
    # for c in range(len(cb_ticks)):
    #     cb_ticks_1f[c] = format(cb_ticks[c], '.1f')
    cb.set_ticklabels(cb_ticks)
    # plt.tight_layout()
    plt.grid(True, linestyle='--')
    plt.show()

def inter_polate(x, y, z, x_lower, x_upper, y_lower, y_upper):
    newfunc = interpolate.interp2d(x, y, z, kind='cubic')
    xnew = np.linspace(x_lower, x_upper, 50)
    ynew = np.linspace(y_lower, y_upper, 50)#linspce默认包含最后一个点, linspace是分多少数，而arange是每段间隔是多少
    fnew = newfunc(xnew, ynew)
    return xnew, ynew, fnew


for i in range(len(data_plot)):
    xticks = np.linspace(x_lower, x_upper, 8, dtype=np.int32)
    yticks = np.linspace(y_lower, y_upper, 6, dtype=np.int32)

    z_min = np.min(data_plot[i])
    z_max = np.max(data_plot[i])

    if z_max <= 4:
        zmajorLocator = MultipleLocator(0.1)  # 将主刻度标签设置为20的倍数
        zmajorFormatter = FormatStrFormatter('%.1f')  # 设置x轴标签文本的格式

    if z_max <= 10 and z_max > 4:
        zmajorLocator = MultipleLocator(0.5)  # 将主刻度标签设置为20的倍数
        zmajorFormatter = FormatStrFormatter('%.1f')  # 设置x轴标签文本的格式

    if z_max > 10 and z_max <= 20:
        zmajorLocator = MultipleLocator(2)  # 将主刻度标签设置为20的倍数
        zmajorFormatter = FormatStrFormatter('%d')  # 设置x轴标签文本的格式

    if z_max > 20:
        zmajorLocator = MultipleLocator(40)  # 将主刻度标签设置为20的倍数
        zmajorFormatter = FormatStrFormatter('%d')  # 设置x轴标签文本的格式
        # ztick_lower = math.floor(z_min / 10) * 10
        # ztick_upper = math.ceil(z_max / 10) * 10
        # y2int_num = 6
        # ztick_unit = math.ceil((ztick_upper - ztick_lower) / y2int_num / 10) * 10
        # ztick_upper_pro = ztick_lower + ztick_unit * (y2int_num + 1)
        # zticks = np.arange(ztick_lower, ztick_upper_pro, ztick_unit)

    x_new, y_new, z_new = inter_polate(time_step, cell_num, data_plot[i], x_lower, x_upper, y_lower, y_upper)
    pltsurf(x_new, y_new, z_new, xticks, yticks, zmajorLocator, zmajorFormatter, object[i])

"""



# 3_MAE_4models
"""

# data = pd.read_excel(r'C:\\Users\TR\Desktop\轨迹预测实验结果.xlsx', sheet_name='error3d_WHLG')
data = pd.read_excel(r'C:\\Users\TR\Desktop\轨迹预测实验结果.xlsx', sheet_name='error3d')
GRU = data._values[0:3, 1:]
LSTM = data._values[3:6, 1:]
SVR = data._values[6:9, 1:]
GM = data._values[9:12, 1:]
ARIMA = data._values[12:15, 2:]
BP = data._values[15:18, 1:]

def plt_3(X, Model, model_name, y_ticks, color):
    plt.figure(figsize=(2.96, 2.22))
    plt.plot(X, Model[0, :], c=color[0], label='Altitude', linewidth=2, linestyle='--', marker='^', markersize=6)
    plt.plot(X, Model[1, :], c=color[1], label='Latitude', linewidth=2, linestyle='--', marker='s', markersize=6)
    plt.plot(X, Model[2, :], c=color[2], label='Longitude', linewidth=2, linestyle='--', marker='d', markersize=6)
    plt.xticks(np.arange(10, 60, 10), np.arange(10, 60, 10), fontsize=10.5)
    plt.yticks(y_ticks, y_ticks, fontsize=10.5)  #如果想改坐标轴标签的内容用这个函数
    plt.xlabel(u'Time step', fontproperties=font_TNM)
    plt.ylabel(u'MAE (m)', fontproperties=font_TNM)

    # plt.legend(loc=2, fontsize=10.5)
    # plt.table()#是添加表格的
    # plt.title(u'Lstm', fontproperties=font_TNM)
    # plt.grid()
    plt.tight_layout()
    # plt.savefig(r'C:\\Users\TR\Desktop\GRU武汉理工\武汉理工GRU\figures' + "\\" + model_name + ".png", dpi=300)
    # plt.savefig(r'D:\轨迹预测\prediction' + "\\" + model_name + ".png", dpi=300)
    plt.show()


Model_name = ["GRU", "LSTM", "SVR", "GM", "ARIMA", "BP"]
# color_set = [sns.xkcd_rgb['candy pink'], sns.xkcd_rgb['light green blue'], sns.xkcd_rgb['dark sky blue']]
color_set = ['lime', 'cyan', 'orange']
Data = [GRU, LSTM, SVR, GM, ARIMA, BP]
for i in range(0, len(Data)):
    if Data[i].shape[1] == 10:
        X1 = np.arange(5, 55, 5)
        for j in range(Data[i].shape[1]):
            # Data[i][:, j] = np.multiply(Data[i][:, j], np.array([Altitude_scalar,
            #                                                      Latitude_scalar,
            #                                                      Longitude_scalar]).T)
            Data[i][:, j] = np.multiply(Data[i][:, j], np.array([Altitude_scalar*Altitude_xishu,
                                                                 Latitude_scalar*Latitude_xishu,
                                                                 Longitude_scalar*Longitude_xishu]).T)

    else:
        X1 = np.arange(10, 55, 5)

    y_min = np.min(Data[i])
    y_max = np.max(Data[i])
    y_lower = math.floor(y_min)
    y_upper = math.ceil(y_max)

    if (y_upper-y_lower) % 4 != 0:
        for k in range(1, 4):
            if (y_upper - y_lower) % 4 == k:
                y_upper = y_upper + (4 - k)
                y_lower_last = y_lower
                y_upper_last = y_upper
                break
    else:
        try:
            y_upper_last
        except NameError:
            var_exists = False
        else:
            var_exists = True
        if var_exists:
            y_lower = y_lower_last
            y_upper = y_upper_last


    y_ticks = np.linspace(y_lower, y_upper, 5)
    plt_3(X1, Data[i], Model_name[i], y_ticks, color_set)

"""


# 1-2-3step
"""

data = pd.read_excel(r'C:\\Users\TR\Desktop\轨迹预测实验结果.xlsx', sheet_name='4model_123')
GRU = data._values[0:3, 1:]
LSTM = data._values[3:6, 1:]
SVR = data._values[6:9, 1:]
GM = data._values[9:12, 1:]
BP = data._values[12:15, 1:]
ARIMA = data._values[15:18, 1:]  # arima不需要反归一化


Altitude = data._values[[0, 3, 12], 1:]*Altitude_scalar
Latitude = data._values[[1, 4, 13], 1:]*Latitude_scalar
Longitude = data._values[[2, 5, 14], 1:]* Longitude_scalar # 这个没有系数换算

Altitude = np.vstack((Altitude, data._values[15, 1:]))
Latitude = np.vstack((Latitude, data._values[16, 1:]))
Longitude = np.vstack((Longitude, data._values[17, 1:]))

model_ticks = ['GRU', 'LSTM', 'BP', 'ARIMA']
AL_color = ['lightgreen', 'limegreen', 'darkgreen']
LA_color = ['skyblue', 'c', 'teal']
LO_color = ['gold', 'orange', 'darkorange']


def bar123(data_set, model_tick, color, name):
    fig = plt.figure(figsize=(5.92, 2.22))
    ax1 = fig.add_subplot(111)
    bar_width = 0.2
    ax1.bar(np.arange(0, len(model_tick), 1), data_set[:, 0], width=bar_width, color=color[0], label='step1')
    ax1.bar(np.arange(0, len(model_tick), 1)+bar_width, data_set[:, 1], width=bar_width, color=color[1], label='step2', tick_label=model_tick)
    ax1.bar(np.arange(0, len(model_tick), 1)+bar_width*2, data_set[:, 2], width=bar_width, color=color[2], label='step3')

    if np.max(data_set) <= 5:
        ytick_max = 6

    if 5 < np.max(data_set) <= 10:
        ytick_max = 18

    if 10 < np.max(data_set) <= 20:
        ytick_max = 20

    if 20 < np.max(data_set) <= 30:
        ytick_max = 30

    ax1.set_yticks(np.linspace(0, ytick_max, 7))
    ax1.set_yticklabels(np.linspace(0, ytick_max, 7), fontsize=10.5)
    ax1.set_ylabel(u'MAE (m)', fontsize=10.5)
    plt.legend(loc=2)
    plt.savefig(r'D:\轨迹预测\prediction\123' + '\\' + name + '.png')
    plt.show()

bar123(Altitude, model_ticks, AL_color, 'Altitude123')
bar123(Latitude, model_ticks, LA_color, 'Latitude123')
bar123(Longitude, model_ticks, LO_color, 'Longitude123')

"""


# 2-3-4 hidden_layer in 5-30-5 cell size (bar)
"""

data = pd.read_excel(r'C:\\Users\TR\Desktop\轨迹预测实验结果.xlsx', sheet_name='2-3-4hidden layer 5-30-5')
data = data._values

for i in range(0, data.shape[0]):
    if i % 3 == 0:
        data[i] = data[i] * Altitude_scalar
    if (i-1) % 3 == 0:
        data[i] = data[i] * Latitude_scalar
    if (i-2) % 3 == 0:
        data[i] = data[i] * Longitude_scalar



def hidden_layer2_3_4(data, color, xticks):
    fig = plt.figure(figsize=(5.92, 3.0))
    ax1 = fig.add_subplot(111)
    # fig, (ax1) = plt.subplots(figsize=(5.92, 2.22), nrow=1 )
    bar_width = 0.2
    # 先画2层时的图,
    x2 = np.arange(0, len(xticks))
    y2 = data[0, :] + data[1, :] + data[2, :]
    ax1.bar(x2, data[0, :], width=bar_width, color=color[0], label='Altitude')
    ax1.bar(x2, data[1, :], width=bar_width, bottom=data[0, :], color=color[1],
            label='Latitude')
    ax1.bar(x2, data[2, :], width=bar_width, bottom=data[1, :], color=color[2],
            label='Longitude')

    # 给顶部加上text代表层数
    # for a, b in zip(x2, y2):
    a = x2[0]
    b = y2[0]
    plt.text(a, b-1, str(2), ha='center', va='bottom', fontsize=10.5)

    # 在width后画3层的图，把tick_label放在这里，居中, 后面的就不需要加label了
    x5 = np.arange(0, len(xticks)) + bar_width
    y5 = data[3, :] + data[4, :] + data[5, :]
    ax1.bar(x5, data[3, :], width=bar_width, color=color[0], tick_label=xticks)
    ax1.bar(x5, data[4, :], width=bar_width, bottom=data[3, :], color=color[1])
    ax1.bar(x5, data[5, :], width=bar_width, bottom=data[4, :], color=color[2])

    # for a, b in zip(x5, y5):
    a = x5[0]
    b = y5[0]
    plt.text(a, b-1, str(3), ha='center', va='bottom', fontsize=10.5)

    # 在2倍width后画4层的图
    x8 = np.arange(0, len(xticks)) + 2 * bar_width
    y8 = data[6, :] + data[7, :] + data[8, :]
    ax1.bar(x8, data[6, :], width=bar_width, color=color[0])
    ax1.bar(x8, data[7, :], width=bar_width, bottom=data[6, :], color=color[1])
    ax1.bar(x8, data[8, :], width=bar_width, bottom=data[7, :], color=color[2])

    # for a, b in zip(x8, y8):
    a = x8[0]
    b = y8[0]
    plt.text(a, b-1, str(4), ha='center', va='bottom', fontsize=10.5)

    # 加一点细节
    ax1.set_yticks(np.linspace(0, 15, 6))
    ax1.set_yticklabels(np.linspace(0, 15, 6), fontsize=10.5)
    ax1.set_xlabel(u'Cell size', fontsize=10.5)
    ax1.set_ylabel(u'MAE (m)')
    # label放一列会遮盖
    plt.legend(loc=1, ncol=3)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(r'C:\\Users\TR\Desktop\GRU武汉理工\武汉理工GRU\figures\hidden2-3-4.png', dpi=400)
    plt.show()

x_ticks = []
for i in range(0, data.shape[1]):
    if i == data.shape[1] - 1:
        x_ticks.append(50)
    else:
        x_ticks.append((i+1) * 5)

color_set = [sns.xkcd_rgb['candy pink'], sns.xkcd_rgb['light green blue'], sns.xkcd_rgb['dark sky blue']]

hidden_layer2_3_4(data, color_set, x_ticks)

"""

# hidden_size3 and cell size 5_30
"""

data = pd.read_excel(r'C:\\Users\TR\Desktop\轨迹预测实验结果.xlsx', sheet_name='hidden3_cell5_50')
data = data._values[:, 1:]
for i in range(0, data.shape[0]):
    if i % 3 == 0:
        data[i] = data[i] * Altitude_scalar
    if (i-1) % 3 == 0:
        data[i] = data[i] * Latitude_scalar
    if (i-2) % 3 == 0:
        data[i] = data[i] * Longitude_scalar

def hidden_layer3(data, color, xticks, model_name):
    fig = plt.figure(figsize=(5.92, 3.0))
    ax1 = fig.add_subplot(111)
    # fig, (ax1) = plt.subplots(figsize=(5.92, 2.22), nrow=1 )
    bar_width = 0.2
    # 先画3层时高度的图,
    x2 = np.arange(0, len(xticks))
    ax1.bar(x2, data[0, :], width=bar_width, color=color[0], label='Altitude')

    # 在width后画3层的纬度图，把tick_label放在这里，居中, 后面的就不需要加label了
    x5 = np.arange(0, len(xticks)) + bar_width
    ax1.bar(x5, data[1, :], width=bar_width, color=color[1], tick_label=xticks, label='Latitude')

    # 在2倍width后画3层的经度图
    x8 = np.arange(0, len(xticks)) + 2 * bar_width
    ax1.bar(x8, data[2, :], width=bar_width, color=color[2], label='Longitude')

    # 加一点细节
    ax1.set_yticks(np.linspace(0, 8, 5))
    ax1.set_yticklabels(np.linspace(0, 8, 5), fontsize=10.5)
    ax1.set_xlabel(u'Cell size', fontsize=10.5)
    ax1.set_ylabel(u'MAE (m)')
    # label放一列会遮盖
    plt.legend(loc=1, ncol=3)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(r'D:\轨迹预测\prediction' + "\\" + model_name + ".png", dpi=400)
    plt.show()

x_ticks = []
for i in range(0, data.shape[1]):
    if i >= data.shape[1] - 2:
        if i == data.shape[1] - 2:
            x_ticks.append(50)
        else:
            x_ticks.append(100)
    else:
        x_ticks.append((i+1) * 5)

color_set = ['lime', 'cyan', 'orange']
model_name = ['step5_hidden3', 'step40_hidden3']
for step_num in range(2):
    data_sub = data[step_num*3:(step_num+1)*3, :]
    hidden_layer3(data_sub, color_set, x_ticks, model_name[step_num])

"""





# 5-30size,1:11:2 layer num
"""

data = pd.read_excel(r'C:\\Users\TR\Desktop\轨迹预测实验结果.xlsx', sheet_name='hidden_paramer')
data = data._values

for i in range(0, data.shape[0]):
    if i % 4 == 0:
        data[i] = data[i] * Altitude_scalar
    if (i-1) % 4 == 0:
        data[i] = data[i] * Latitude_scalar
    if (i-2) % 4 == 0:
        data[i] = data[i] * Longitude_scalar



def plot_5_30_234cell(x, y1, y2, y3, y4, color_set, y1_ticks, y2_ticks, title):
    fig = plt.figure(figsize=(2.96, 2.22))
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, color=color_set[0], linestyle='--', linewidth=1.25, marker='^', markersize=6.0)
    ax1.plot(x, y2, color=color_set[1], linestyle='--', linewidth=1.25, marker='s', markersize=6.0)
    ax1.plot(x, y3, color=color_set[2], linestyle='--', linewidth=1.25, marker='d', markersize=6.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x, fontsize=10.5)
    ax1.set_yticks(y1_ticks)
    ax1.set_yticklabels(y1_ticks, fontsize=10.5)
    ax1.set_xlabel(u'Number of hidden layers', fontsize=10.5)
    ax1.set_ylabel(u'MAE (m)', fontsize=10.5)
    ax1.grid(False)
    # ax2 = ax1.twinx()
    # ax2.plot(x, y4, color=color_set[3], linestyle='--', linewidth=1.25, marker='o', markersize=6.0)
    # ax2.set_yticks(y2_ticks)
    # ax2.set_yticklabels(y2_ticks, fontsize=10.5)
    # ax2.set_ylabel(u'Training time (sec)', fontsize=10.5)
    plt.tight_layout()
    plt.savefig(r'C:\\Users\TR\Desktop\GRU武汉理工\武汉理工GRU\figures' + '\\' + 'hlayer_number' + title + '.png', dpi=400)
    plt.show()

x_range = np.arange(1, 11, 2)
color = [sns.xkcd_rgb['candy pink'], sns.xkcd_rgb['light green blue'], sns.xkcd_rgb['dark sky blue'], 'putty']
file_name = []
for c in range(1, 7):
    cell_size = 5*c
    file_name.append('Cell' + str(cell_size))


for i in range(len(file_name)):
    set_start = i * 4
    set_end1 = (i * 4) + 3
    # set_end2 = (i + 1) * 6

    mae_lower = 0
    # mae_upper_set = np.zeros(2)
    # space = [2, 3]
    # for u in range(2):
    #     unit = space[u]
    #     for k in range(1, 11):
    #         diff = k * unit - np.max(data[set_start:set_end1])
    #         if 0 < diff <= unit:
    #             mae_upper_set[u] = k * unit
    #             break
    # if mae_upper_set[0] >= mae_upper_set[1]:
    #     mae_upper = mae_upper_set[0]
    #     mae_unit = space[0]
    # else:
    #     mae_upper = mae_upper_set[1]
    #     mae_unit = space[1]

    space = [3]
    unit = space[0]
    mae_upper_set = np.zeros(1) #这样写虽然没意义，但是可以使得yticklabels是一位小数
    for k in range(1, 11):
        diff = k * unit - np.max(data[set_start:set_end1])
        if 0 < diff <= unit:
            mae_upper_set[0] = k * unit
            break

    mae_upper = mae_upper_set[0]
    mae_unit = space[0]

    y1ticks = np.arange(mae_lower, mae_upper[0]+mae_unit, mae_unit)

    # data_time = np.zeros(len(x_range))
    # for col in range(0, len(x_range)):
    #     data_time[col] = np.max(data[set_end1:set_end2, col])
    #
    # time_lower = math.floor(data_time / 10) * 10
    # time_upper = math.ceil(data_time / 10) * 10
    # if time_upper <= 90:
    #     y2int_num = 4
    #
    # if 90 < time_upper <= 130:
    #     y2int_num = 3
    #
    # if time_upper > 130:
    #     y2int_num = 6
    # time_unit = math.ceil((time_upper - time_lower)/y2int_num/10)*10
    # time_upper_pro = time_lower + time_unit*(y2int_num+1)
    # y2ticks = np.arange(time_lower, time_upper_pro, time_unit)

    plot_5_30_234cell(x_range, data[set_start,:], data[set_start+1,:], data[set_start+2,:], 0,
                   color, y1ticks, 0, file_name[i])
"""

# box plot for MAE
"""

CwRnn_data = pd.read_csv(r'D:\轨迹预测\prediction\CWRNN\error5_50.csv')
CwRnn_data_mae = CwRnn_data._values[:, 1]
Gru_data = pd.read_csv(r'D:\轨迹预测\prediction\GRU\error5_50.csv')
Gru_data_mae = Gru_data._values[:, 2]
Lstm_data = pd.read_csv(r'D:\轨迹预测\prediction\LSTM\error5_50.csv')
Lstm_data_mae = Lstm_data._values[:, 2]


fig = plt.figure(figsize=(5.96, 2.22))
ax1 = fig.add_subplot(111)
#  fig, (ax1) = plt.subplots(figsize=(5.96, 2.22), nrows=1)
labels = ['GRU', 'LSTM', 'CWRNN']
plt.boxplot([Gru_data_mae, Lstm_data_mae, CwRnn_data_mae], labels=labels,
            showmeans=True,
            showfliers=True,
            flierprops={'markeredgecolor': 'pink','marker': '.', 'markersize':1.0}
            )
plt.show()

"""


# train loss plot
# GRU_train = pd.read_excel(r'C:\\Users\TR\Desktop\轨迹预测实验结果.xlsx',
#                           sheet_name='GRU_train_time')
# GRU_time = np.mean(GRU_train._values[:, 1::2], 1)
# GRU_mse = np.mean(GRU_train._values[:, 2::2], 1)
# LSTM_train = pd.read_excel(r'C:\\Users\TR\Desktop\轨迹预测实验结果.xlsx',
#                           sheet_name='LSTM_train_time')
# LSTM_time = np.mean(LSTM_train._values[:, 1::2], 1)
# LSTM_mse = np.mean(LSTM_train._values[:, 2::2], 1)

GRU_train = pd.read_csv(r'D:\轨迹预测\prediction\GRU\train_loss_mean0.csv')
GRU_time = GRU_train._values[:, 1]
GRU_mse = GRU_train._values[:, 2]
LSTM_train = pd.read_csv(r'D:\轨迹预测\prediction\LSTM\train_loss_mean0.csv')
LSTM_time = LSTM_train._values[:, 1]
LSTM_mse = LSTM_train._values[:, 2]

def plot_train_time_loss(X1, Y1, X2, Y2, color, marker, labels):
    fig = plt.figure(figsize=(2.96, 2.22))
    ax = fig.add_subplot(111)
    ax.plot(X1, Y1, color=color[0], linestyle='-', linewidth=1.25, marker=marker[0], markersize=2.0, label=labels[0])
    ax.plot(X2, Y2, color=color[1], linestyle='-', linewidth=1.25, marker=marker[1], markersize=2.0, label=labels[1])
    ceil_x = math.ceil(max(np.max(X1), np.max(X2)))
    ax.set_xticks(np.linspace(0, ceil_x, ceil_x+1))
    ax.set_xticklabels(np.linspace(0, ceil_x, ceil_x+1), fontsize=10.5)
    ax.set_xlabel(u'Training Time (s)')
    ax.set_ylabel(u'MSE')
    plt.legend()
    plt.tight_layout()
    plt.show()


gru_time = np.zeros(31, dtype='float32')
gru_mse = np.zeros(31, dtype='float32')

for i in range(gru_mse.shape[0]):
    if i % 2 == 0:
        gru_time[i] = GRU_time[i*10]
        gru_mse[i] = GRU_mse[i*10]
    else:
        gru_time[i] = GRU_time[i*10]
        gru_mse[i] = np.mean(GRU_mse[(i-1)*10 + 1:(i+1)*10])


lstm_time = np.zeros(31, dtype='float32')
lstm_mse = np.zeros(31, dtype='float32')

for i in range(lstm_mse.shape[0]):
    if i % 2 == 0:
        lstm_time[i] = LSTM_time[i*10]
        lstm_mse[i] = LSTM_mse[i*10]
    else:
        lstm_time[i] = LSTM_time[i*10]
        lstm_mse[i] = np.mean(LSTM_mse[(i-1)*10 + 1:(i+1)*10])

color_set = ['green', 'red']
marker_set = ['v', 's']
labels_set = ['GRU_train', 'LSTM_train']

plot_train_time_loss(gru_time, gru_mse, lstm_time, lstm_mse, color_set, marker_set, labels_set)
plot_train_time_loss(GRU_time[0:100], GRU_mse[0:100], LSTM_time[0:100], LSTM_mse[0:100], color_set, marker_set, labels_set)
