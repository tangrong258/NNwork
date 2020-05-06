#对于 METAR报进行等级划分
import csv
import pandas as pd
import pickle
import os
import numpy as np
with open(r"D:\ZSNJAP01\METAR\MEZSNJ\m1to4.csv", "r") as f:
    reader = csv.reader(f)
    column = [row[4] for row in reader]#row[4]就是取一行的第5个元素，总的就是取reader第5列
# print(column)

gra = ["BR","SH","RA","TS","SN"]

W= []
for  i in range(len(column)):
    if column[i].count(gra[4]) and column[i].count("ZSNJ")!= 1 :
        W.append(6)
    elif column[i].count(gra[3]):
        W.append(5)
    elif column[i].count(gra[2]) and column[i].count(gra[1])!= 1 and column[i].count(gra[3])!=1:
        W.append(4)
    elif column[i].count(gra[1]):
        W.append(3)
    elif column[i].count(gra[0]):
        W.append(2)
    else:
        W.append(1)
del W[0]
# dataframe = pd.DataFrame(W)
# dataframe.to_csv(r"C:\Users\tangrong\Desktop\METAR\metartest04.csv", index=False, sep=',')
for i in range(1, 7):
    space = i * 10
    n = int(60/space)
    w = np.zeros(len(W) * n)
    for j in range(len(W)):
        w[n * j: n * (j+1)] = W[j]
    howlong = str(space) + "min"
    root = r"D:\ZSNJAP01\flight\delaydata" + "\\" + howlong
    b = os.path.exists(root)
    if b:
        print("path exist")
    else:
        os.makedirs(root)
    with open(root + "\\" + "weather5.pkl",'wb') as f:
        pickle.dump(w, f)