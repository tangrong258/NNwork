import xlrd
from datetime import date, datetime
import csv
import pandas as pd
import numpy as np
# 文件
# ExcelFile = xlrd.open_workbook(r"C:\Users\tangrong\Desktop\METARO5.xls")
sheet = np.loadtxt(open(r"C:\Users\tangrong\Desktop\METAR05.csv", "rb"), delimiter=",", skiprows=0)
# 获取目标EXCEL文件sheet名C:\Users\tangrong\Desktop
# print(ExcelFile.sheet_names())
# 若有多个sheet，则需要指定读取目标sheet例如读取sheet2
# sheet2_name=ExcelFile.sheet_names()[1]
# 获取sheet内容【1.根据sheet索引2.根据sheet名称】
# sheet=ExcelFile.sheet_by_index(1)
# sheet = ExcelFile.sheet_by_name('sheet1')
# 打印sheet的名称，行数，列数
# print(sheet.name, sheet.nrows, sheet.ncols)
# 获取整行或者整列的值
# rows = sheet.row_values(2)  # 第三行内容
cols1 = sheet.col_values(1)  # 第二列内容
cols2 = sheet.col_values(2)

# # 获取单元格内容
# print(sheet.cell(1, 0).value.encode('utf-8'))
# print(sheet.cell_value(1, 0).encode('utf-8'))
# print (sheet.row(1)[0].value.encode('utf-8'))
# # 打印单元格内容格式
# print(sheet.cell(1, 0).ctype)
RA = "RA"
W= []
for  i in range(len(cols2)):
    if cols2(i).count(RA):
        W[i] = 1
# 字典中的key值即为csv中列名
dataframe = pd.DataFrame({'dengji':W})#传入的值不能直接是value，而得是list
# #两种写法
#columns = ["URL", "predict", "score"]
#dt = pd.DataFrame(result_list, columns=columns)
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv(r"C:\Users\tangrong\Desktop\metartest.csv", index=False, sep=',')