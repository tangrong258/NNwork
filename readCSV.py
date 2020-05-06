# import tensorflow as tf
# import numpy as np
# import pandas as pd
#
# # 从CSV文件中读取数据，并返回2个数组。分别是自变量x和因变量y。方便TF计算模型。
# def zc_read_csv():
#     zc_dataframe = pd.read_csv(r"C:\Users\tangrong\Desktop\ZSNJ201805.csv", encoding="unicode_escape",sep=",")
#     x = []
#     y = []
#     for zc_index in zc_dataframe.index:
#         zc_row = zc_dataframe.loc[zc_index]
#         x.append(zc_row["B"])
#         y.append(zc_row["L"])
#     return (x,y)
#
# x, y = zc_read_csv()
#
# print("x:",x)
# print("y:",y)
import csv
import numpy as np

# 读取csv
csvFile = open(r'C:\Users\tangrong\Desktop\ZSNJ201805.csv', "r")
reader = csv.reader(csvFile)
list=[]
list1=[]
list2=[]
#全部读进来，再操作
print(reader)
for row in reader:
    # list1 += [row[2]]
    # list2 += [row[3]]
    list += [row[2],row[3]]
# print("list1:",list1)
# print("list2:",list2)
#list = [list1 ,list2]
print("list:",list)
# 为什么只能读取1列
# list1 += [row[3] for row in reader ]
# print(list1)
# 为什么读出的都是单个字符
# for row in reader:
#     list1 += row[2]
#     list2 += row[3]
# list = list1 + list2
# print(list)
#变维操作，
b = np.array(list).reshape(31,48,2)
print(b)


