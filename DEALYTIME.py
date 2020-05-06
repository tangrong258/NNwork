#统计每隔一小时各个点的流量

import csv
import time
import pandas as pd
import numpy as np
import pickle
import os

# with open(r"D:\ZSNJAP01\flight\all.csv",'r')as f:
#     reader = csv.reader(f)
#     column1 = [row[3] for row in reader]#进入点
#     column2 = [row[4] for row in reader]#离开
#     column3 = [row[5] for row in reader]#进入时间
#     column4 = [row[6] for row in reader]#间隔
column =[[] for i in range(4)]#4个含有list的列表
for i in range(3,7):
    with open(r"D:\ZSNJAP01\flight\flightinfo\m1to4.csv", 'r')as f:
        reader = csv.reader(f)
        column[i-3]= [row[i] for row in reader]

# with open(r"D:\ZSNJAP01\METAR\MEZSNJ\all.csv",'r')as f:
#     reader = csv.reader(f)
#     timesp = [row[1] for row in reader]


enter =['OREVO','GOSRO','NJ120','NJ403']
exit = ['NJ214','NJ218','NJ106','NJ406','NJ407']

#计算各个航段的平均飞行时间
#先分类
#两种计算方式，如果将数据列放在外循环，20中组合放在内循环，判断正确跳出循环，加快循环
#如果数据列在内循环，你没法跳，所以应该将1对多的放在里面，一个组合包含多个序列里面的数据
lenc = len(column[0])
kind = [[[]for j in range(5)]for i in range(4)]#4行5列
for k in range(lenc):
    for i in range(4):
        for j in range(5):
            if column[0][k].count(enter[i])  and column[1][k].count(exit[j]):
                kind[i][j].append(column[3][k])
                break
#因为在i = 0的时候，j肯定会执行一次，如果在i循环里再加入一个break，那么久只能得到i=0的值
#kind不能转换为数组，因为维度不一样
#kind 里面会有（i,j）的组合为空，需要补为0，以便求解分位数
kindt = [[[]for j in range(5)]for i in range(4)]#飞行时间
for i in range(len(kind)):
    for j in range(len(kind[i])):
        if len(kind[i][j]) == 0:
            kind[i][j].append(0)
        for k in range(len(kind[i][j])):
            kindt[i][j].append(float(kind[i][j][k]))
#转换为浮点型

perkind  = np.zeros((4,5))
for i in range(4):
    for j in range(5):
        perkind[i][j] = np.percentile(kindt[i][j],20)

dlytime = [[] for i in range(1)]
m = perkind.shape[0]
n = perkind.shape[1]#记住，数越大，代表越靠里
for k in range(lenc):
    for i in range(m):
        for j in range(n):
            if column[0][k].count(enter[i]) and column[1][k].count(exit[j]):
                dtime = float(column[3][k]) - perkind[i][j]
                if dtime > 0:
                    dlytime[0].append(dtime)
                else:
                    dlytime[0].append(0)
                break

#dlytime得有70484个数据，dtype = float

#计算分类的平均延误时间，要不要算

column = column + dlytime


#将日期格式转化为时间戳

enternum = [[] for i in range(lenc)]
exitnum = [[] for i in range(lenc)]
for i in range (0,lenc):
    enterstard = time.strptime(column[2][i], "%Y/%m/%d %H:%M")
    enternum[i]= time.mktime(enterstard)
    exitnum[i] = enternum[i] + float(column[3][i]) * 60

# create time squence
howlong = "15min"
timesq1 = pd.date_range(start="2018/1/01 00:00",end = "2018/4/01 00:00",freq=howlong)
timesq2 = pd.date_range(start="2018/5/01 00:00",end = "2018/5/31 00:00",freq=howlong)


timespace = []

for i in range(0,len(timesq1)):
    timespace.append(str(timesq1[i]))
for i in range(0,len(timesq2)):
    timespace.append(str(timesq2[i]))

lent =len(timesq1) + len(timesq2)
timespnum = [[] for i in range(lent)]
for i in range(0, lent):
    timespstard = time.strptime(timespace[i], "%Y-%m-%d %H:%M:%S")
    timespnum[i] = time.mktime(timespstard)
#将时间戳变为日期格式
# stard=time.localtime(num)
# dandt=time.strftime('%Y-%m-%d %H:%M:%S',stard)



#循环结构可能改一下，计算速度会提升，每一个enter点都将所有的航班遍历一遍，太慢了，
#应该改为在一个时间段内将4个enter的flow都算出来
# timespdown = [[] for i in range(len(timespnum)-1)]
# timespup =[[] for i in range(len(timespnum)-1)]
# enterspot =[[] for i in range(len(enter))]
# exitspot =[[] for i in range(len(exit))]
# for i in range(0,len(enter)):
#     for j in range (0,len(timespnum)-1):
#         enterflow = 0
#         timespdown[j] = timespnum[j]
#         timespup[j] = timespnum[j+1]
#         for k in range (0,len(enternum)):
#             if enternum[k] >= timespdown[j] and enternum[k] < timespup[j]:
#                 if column[0][k].count(enter[i]):
#                     enterflow = enterflow + 1
#             if enternum[k] >= timespup[j]:
#                 break
#         print(enterflow)
#         enterspot[i].append(enterflow)
# for i in range(0, len(exit)):
#     for j in range(0, len(timespnum) - 1):
#         exitflow = 0
#         timespdown[j] = timespnum[j]
#         timespup[j] = timespnum[j + 1]
#         for k in range(0, len(exitnum)):
#             if exitnum[k] >= timespdown[j] and exitnum[k] < timespup[j]:
#                 if column[1][k].count(exit[i]):
#                     exitflow = exitflow + 1
#             if exitnum[k] >= timespup[j]:
#                 break
#         print(exitflow)
#         exitspot[i].append(exitflow)
# spot = enterspot + exitspot

#这个方法，当重新开始一个时间段timespnum，column又要从头遍历
# for j in range(0,len(timespnum)-1):
#     for k in range(0,len(enternum)):
#         if enternum[k] >= timespnum[j] and enternum[k] < timespnum[j+1]:
#             for i in range(enter):
#                 if column[0][k].count(enter[i]):
#                     enterflow(i) = enterflow(i) + 1
#                     break
#         if enternum[k] >= timespup[j]:
#             break


#把需要遍历的放在外面,因为timespnum的序列也长，所以一个时间段内需要做更多的事情
enterflow = np.zeros((len(timespnum)-1,len(enter)))
exitflow = np.zeros((len(timespnum)-1,len(exit)))
linedlyt = np.zeros(((len(timespnum)-1,len(enter),len(exit))))
linedlyf = np.zeros(((len(timespnum)-1,len(enter),len(exit))))
linez = np.zeros(((len(timespnum)-1,len(enter),len(exit))))
linef = np.zeros(((len(timespnum)-1,len(enter),len(exit))))
lineavgt = np.zeros(((len(timespnum)-1,len(enter),len(exit))))
for k in range(0,len(enternum)):
    for j in range(0,len(timespnum)-1):
        if enternum[k] >= timespnum[j] and enternum[k] < timespnum[j+1]:
            for p in range(len(enter)):
                if column[0][k].count(enter[p]):
                    enterflow[j,p] = enterflow[j,p] + 1
                    for q in range(0,len(exit)):
                        if column[1][k].count(exit[q]):
                            linef[j,p,q] = linef[j,p,q] + 1 #总架次
                            linez[j,p,q] = linez[j,p,q] + float(column[3][k])#总时间
                            lineavgt[j,p,q] = linez [j,p,q] / linef[j,p,q]#平均飞行时间
                            if column[4][k] > 0:
                                linedlyf[j,p,q] = linedlyf[j,p,q] + 1#延误总架次
                                linedlyt[j,p,q] = linedlyt[j,p,q] + column[4][k]  # 延误总时间
                            break
                    break

        if exitnum[k] >= timespnum[j] and exitnum[k] < timespnum[j+1]:
            for m in range(len(exit)):
                if column[1][k].count(exit[m]):
                    exitflow[j,m] = exitflow[j,m] + 1
                    break

root = r"D:\ZSNJAP01\flight\delaydata" + "\\"+ howlong
b = os.path.exists(root)
if b :
    print("path exist")
else:
    os.makedirs(root)


with open(root + "\\" + "linedlyt.pkl",'wb') as f:
    pickle.dump(linedlyt,f)
with open(root + "\\" + "linedlyf.pkl",'wb') as f:
    pickle.dump(linedlyf,f)
with open(root + "\\" + "linez.pkl",'wb') as f:
    pickle.dump(linez,f)
with open(root + "\\" + "linef.pkl",'wb') as f:
    pickle.dump(linef,f)
with open(root + "\\" + "lineavgt.pkl",'wb') as f:
    pickle.dump(lineavgt,f)
    f.close()

spot = np.hstack((enterflow, exitflow))#vstack增加行,hstack增加列
with open(root + "\\" + "spot.pkl",'wb') as f:
    pickle.dump(spot, f)
# df = pd.DataFrame(linedlyt,index = enter + exit )
# df = df.T
# df.to_csv(r"D:\ZSNJAP01\flight\delaydata\spot.csv", index=False, sep=',')
