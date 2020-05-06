from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties as FP
import pandas as pd
import numpy as np
plt.rc('font', family='Times New Roman')
font_TNM = FP(fname=r"C:\Windows\Fonts\times.ttf", size=10.5)


data = pd.read_excel(r"D:\轨迹预测\prediction\error3d.xlsx", sheet_name='Sheet2')
LSTM = data._values[0:3, 1:data._values.shape[1]+1]
GRU = data._values[3:6, 1:data._values.shape[1]+1]
SVR = data._values[6:9, 1:data._values.shape[1]+1]
GM = data._values[9:12, 1:data._values.shape[1]+1]

def plt_3(X, Model, model_name):
    plt.figure(figsize=(2.96, 2.22))
    plt.plot(X, Model[0, :]*78, c='lime', label='Altitude', linewidth=2, linestyle='--', marker='^', markersize=6)
    plt.plot(X, Model[1, :]*163, c='cyan', label='Latitude', linewidth=2, linestyle='--', marker='s', markersize=6)
    plt.plot(X, Model[2, :]*154, c='orange', label='Longitude', linewidth=2, linestyle='--', marker='d', markersize=6)
    plt.xticks(fontsize=10.5)
    plt.yticks(fontsize=10.5)
    # plt.yticks(np.arange(0, 0.1, 0.02), np.arange(0, 0.1, 0.02), fontsize=10.5)#如果想改坐标轴标签的内容用这个函数
    plt.xlabel(u'Timestep', fontproperties=font_TNM)
    plt.ylabel(u'MAE (m)', fontproperties=font_TNM)
    # plt.legend(loc=2, fontsize=10.5)
    # plt.table()#是添加表格的
    # plt.title(u'Lstm', fontproperties=font_TNM)
    # plt.grid()
    plt.tight_layout()
    plt.savefig(r"D:\轨迹预测\prediction" + "\\" + model_name + ".png", dpi=300)
    plt.show()

def plt_4(X, Model, model_name):
    plt.figure(figsize=(2.96, 2.22))
    plt.plot(X, Model[0, :]*78, color='lime', marker='^', linestyle='--', label='Altitude', linewidth=2, markersize=6)
    plt.plot(X, Model[1, :]*163, color='cyan', marker='s', linestyle='--', label='Latitude', linewidth=2, markersize=6)
    plt.plot(X, Model[2, :]*154, color='orange', marker='d', linestyle='--',  label='Longitude', linewidth=2, markersize=6)
    plt.xticks(fontsize=10.5)
    plt.yticks(fontsize=10.5)
    # plt.yticks(np.arange(0, 0.5, 0.1), [0, '%.2f'%0.10, '%.2f'%0.20, '%.2f'%0.30, '%.2f'%0.40],  fontsize=10.5)#如果想改坐标轴标签的内容用这个函数
    plt.xlabel(u'Timestep', fontproperties=font_TNM)
    plt.ylabel(u'MAE (m)', fontproperties=font_TNM)
    # plt.legend(loc=2, fontsize=10.5)
    # plt.table()#是添加表格的
    # plt.title(u'Lstm', fontproperties=font_TNM)
    # plt.grid()
    plt.tight_layout()
    plt.savefig(r"D:\轨迹预测\prediction" + "\\" + model_name + ".png", dpi=300)
    plt.show()


Model_name = ["LSTM", "GRU", "SVR", "GM"]
X1 = np.arange(5, 45, 5)
Data = [LSTM, GRU, SVR, GM]
for i in range(0, len(Data)-1):
        plt_3(X1, Data[i], Model_name[i])

for i in range(3, len(Data)):
        plt_4(X1, Data[i], Model_name[i])
