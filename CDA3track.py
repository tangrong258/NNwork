
import numpy as np
import csv
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties as FP
plt.rc('font', family='Times New Roman')
font_TNM = FP(fname=r"C:\Windows\Fonts\times.ttf", size=10.5)

root = r'C:\Users\tangrong\Desktop\吴迪\新建文件夹 (2)\新建文件夹 (2)'
data =[[[] for j in range(3)] for i in range(2)]


for i in range(0, 2):
    for j in range(0, 3):
        sub = 'Flight'+str(i+13)+'_'+'ClusterTraj'+'.csv'
        file = open(root+'\\'+sub)
        reader = csv.reader(file)
        for row in reader:
            data[i][j] = [row[j] for row in reader]

# data = np.array(data, dtype=np.float32)

data_clus1 = np.array(data[0], dtype=np.float32)
data_clus2 = np.array(data[1], dtype=np.float32)
data_opt = [[]for i in range(3)]
for j in range(0, 3):
    file = open(root + '\\' + 'opt.csv')
    reader = csv.reader(file)
    for row in reader:
        data_opt[j] = [row[j] for row in reader] #row[j]指的是row这一行第j个

data_opt = np.array(data_opt, dtype=np.float32)

origin = np.array([116.0, 38.8, 0.0], dtype=np.float32)
# arr_point = np.array([[116.2, 39.04, 5400], [116.41, 39.13, 5700], [116.57, 39.19, 5400], [116.78, 39.16, 4200]]).T
for i in range(data_clus1.shape[1]):
    data_clus1[0, i] = (data_clus1[0, i] - origin[0]) * (111*math.cos(40/180*math.pi))
    data_clus1[1, i] = (data_clus1[1, i] - origin[1]) * 111
    data_clus1[2, i] = (data_clus1[2, i] - origin[2])/1000

for i in range(data_clus2.shape[1]):
    data_clus2[0, i] = (data_clus2[0, i] - origin[0]) * (111*math.cos(40/180*math.pi))
    data_clus2[1, i] = (data_clus2[1, i] - origin[1]) * 111
    data_clus2[2, i] = (data_clus2[2, i] - origin[2])/1000

for i in range(data_opt.shape[1]):
    data_opt[0, i] = (data_opt[0, i] - origin[0]) * (111*math.cos(40/180*math.pi))
    data_opt[1, i] = (data_opt[1, i] - origin[1]) * 111
    data_opt[2, i] = (data_opt[2, i] - origin[2])/1000

linew = 1.5
fig1 = plt.figure(figsize=(3.8, 3.6))
# ax1 = plt.axes(projection='3d')
ax1 = fig1.add_subplot(111, projection='3d')

# point_color = ['red', 'orange', 'yellow', 'green', 'cyan', 'magenta']
# line_color = ['red', 'orange', 'yellow', 'green', 'cyan', 'magenta']
#
# for i in range(len(data)):
#     ax1.scatter3D(data[i, 0], data[i, 1], data[i, 2], c=point_color[i], s=4.0)  # 绘制散点图
#     ax1.plot3D(data[i, 0], data[i, 1], data[i, 2], line_color[i], linewidth='0.5')  # 绘制空间曲线
#
# ax1.scatter3D(arr_point[0], arr_point[1], arr_point[2], c=["b", "r", "m", "g"], s=4.0)

# ax1.scatter3D(data_clus1[0], data_clus1[1], data_clus1[2], c='orchid', s=4.0)
# ax1.plot3D(data_clus1[0], data_clus1[1], data_clus1[2], 'orchid', linestyle='-', linewidth=linew, label='Cluster1')

# ax1.scatter3D(data_clus2[0], data_clus2[1], data_clus2[2], c='orange', s=4.0)
# ax1.plot3D(data_clus2[0], data_clus2[1], data_clus2[2], 'orange', linestyle='-', linewidth=linew, label='Cluster2')

# ax1.scatter3D(data_opt[0], data_opt[1], data_opt[2], c='cyan', s=4.0)  # 绘制散点图
ax1.plot3D(data_opt[0], data_opt[1], data_opt[2], 'cyan', linestyle='-',  linewidth=linew, label='Optimize')  # 绘制空间曲线
ax1.set_xticks(np.linspace(10, 50, 5))
ax1.set_xticklabels([10, 20, 30, 40, 50], fontsize=10.5)
ax1.set_yticks(np.linspace(30, 180, 6))
ax1.set_yticklabels(np.arange(30, 210, 30), fontsize=10.5)
ax1.set_zticks(np.linspace(0, 5, 6))
ax1.set_zticklabels(np.arange(0, 6, 1), fontsize=10.5)

# ax1.set_zticklabels([1000, 2000, 3000, 4000, 5000, 6000], fontsize=10.5)
ax1.set_xlabel(u'Longitude(km)', fontsize=10.5)
ax1.set_ylabel(u'Latitude(km)', fontsize=10.5)
ax1.set_zlabel(u'Altitude(m)', fontsize=10.5)
# plt.legend(loc=2, fontsize=10.5)
ax1.grid(True)
# ax1.axis('off')
# plt.savefig(root + "\\" + '3D' + ".png", dpi=300)
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.show()

fig2 = plt.figure(figsize=(2.96, 2.22))
plt.plot(data_clus1[0], data_clus1[2], 'orchid', linestyle='-', label='Cluster1', linewidth=linew)
plt.plot(data_clus2[0], data_clus2[2], 'orange', linestyle='-', label='Cluster2', linewidth=linew)
plt.plot(data_opt[0], data_opt[2], 'cyan', linestyle='-', label='Optimizer', linewidth=linew)
plt.xticks(np.arange(0, 60, 10), fontsize=10.5)
plt.yticks(fontsize=10.5)
plt.xlabel(u'Longitude(km)', fontsize=10.5)
plt.ylabel(u'Altitude(km)', fontsize=10.5)
# plt.legend(loc=2, fontsize=10.5)
plt.grid(True)
plt.savefig(root + "\\" + 'Front_view' + ".png", dpi=300)
plt.show()

fig3 = plt.figure(figsize=(2.96, 2.22))
plt.plot(data_clus1[1], data_clus1[2], 'orchid', linestyle='-', label='Cluster1', linewidth=linew)
plt.plot(data_clus2[1], data_clus2[2], 'orange', linestyle='-', label='Cluster2', linewidth=linew)
plt.plot(data_opt[1], data_opt[2], 'cyan', linestyle='-', label='Optimizer', linewidth=linew)
plt.xticks(np.arange(0, 210, 30), fontsize=10.5)
plt.yticks(fontsize=10.5)
plt.xlabel(u'Latitude(km)', fontsize=10.5)
plt.ylabel(u'Altitude(km)', fontsize=10.5)
# plt.legend(loc=2, fontsize=10.5)
plt.grid(True)
plt.savefig(root + "\\" + 'Right_view' + ".png", dpi=300)
plt.show()

fig4 = plt.figure(figsize=(2.96, 2.22))
plt.plot(data_clus1[0], data_clus1[1], 'orchid', linestyle='-', label='Cluster1', linewidth=linew)
plt.plot(data_clus2[0], data_clus2[1], 'orange', linestyle='-', label='Cluster2', linewidth=linew)
plt.plot(data_opt[0], data_opt[1], 'cyan', linestyle='-', label='Optimizer', linewidth=linew)
plt.xticks(np.arange(0, 60, 10), fontsize=10.5)
plt.yticks(np.arange(0, 210, 30), fontsize=10.5)
plt.xlabel(u'Longitude(km)', fontsize=10.5)
plt.ylabel(u'Latitude(km)', fontsize=10.5)
# plt.legend(loc=2, fontsize=10.5)
plt.grid(True)
plt.savefig(root + "\\" + 'Vertical_view' + ".png", dpi=300)
plt.show()

