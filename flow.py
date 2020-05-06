#统计每隔一小时各个点的流量

import csv
import time
import pandas as pd



# with open(r"D:\ZSNJAP01\flight\all.csv",'r')as f:
#     reader = csv.reader(f)
#     column1 = [row[3] for row in reader]#进入点
#     column2 = [row[4] for row in reader]#离开
#     column3 = [row[5] for row in reader]#进入时间
#     column4 = [row[6] for row in reader]#间隔
column =[[] for i in range(4)]#4列的列表
for i in range(3,7):
    with open(r"D:\ZSNJAP01\flight\all.csv", 'r')as f:
        reader = csv.reader(f)
        column[i-3]= [row[i] for row in reader]

with open(r"D:\ZSNJAP01\MEZSNJ\all.csv",'r')as f:
    reader = csv.reader(f)
    timesp = [row[1] for row in reader]


enter =['OREVO','GOSRO','NJ120','NJ403']
exit = ['NJ214','NJ218','NJ106','NJ406','NJ407']

#将日期格式转化为时间戳
lenc = len(column[2])
enternum = [[] for i in range(lenc)]
exitnum = [[] for i in range(lenc)]
for i in range (0,lenc):
    enterstard = time.strptime(column[2][i], "%Y/%m/%d %H:%M")
    enternum[i]= time.mktime(enterstard)
    exitnum[i] = enternum[i] + float(column[3][i]) * 60

lent =len(timesp)
timespnum = [[] for i in range(lent)]
for i in range (0,lent):
    timespstard = time.strptime(timesp[i], "%Y/%m/%d %H:%M")
    timespnum[i] = time.mktime(timespstard)
#将时间戳变为日期格式
# stard=time.localtime(num)
# dandt=time.strftime('%Y-%m-%d %H:%M:%S',stard)

timespdown = [[] for i in range(len(timespnum)-1)]
timespup =[[] for i in range(len(timespnum)-1)]
enterspot =[[] for i in range(len(enter))]
exitspot =[[] for i in range(len(exit))]
for i in range(0,len(enter)):
    for j in range (0,len(timespnum)-1):
        enterflow = 0
        timespdown[j] = timespnum[j]
        timespup[j] = timespnum[j+1]
        for k in range (0,len(enternum)):
            if enternum[k] >= timespdown[j] and enternum[k] < timespup[j]:
                if column[0][k].count(enter[i]):
                    enterflow = enterflow + 1
            if enternum[k] >= timespup[j]:
                break
        print(enterflow)
        enterspot[i].append(enterflow)
for i in range(0, len(exit)):
    for j in range(0, len(timespnum) - 1):
        exitflow = 0
        timespdown[j] = timespnum[j]
        timespup[j] = timespnum[j + 1]
        for k in range(0, len(exitnum)):
            if exitnum[k] >= timespdown[j] and exitnum[k] < timespup[j]:
                if column[1][k].count(exit[i]):
                    exitflow = exitflow + 1
            if exitnum[k] >= timespup[j]:
                break
        print(exitflow)
        exitspot[i].append(exitflow)#列表才能这样相加,达到合并

spot = enterspot + exitspot

df = pd.DataFrame(spot,index = enter + exit )
df = df.T
df.to_csv(r"C:\Users\tangrong\Desktop\spot.csv", index=False, sep=',')