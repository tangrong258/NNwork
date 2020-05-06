import os
import csv
Folder_Path = r'D:\ZSNJAP01\flight'  # 要拼接的文件夹及其完整路径，注意不要包含中文
SaveFile_Path = r'C:\Users\tangrong\Desktop'  # 拼接后要保存的文件路径
SaveFile_Name = r'all.xls'  # 合并后要保存的文件名

# 修改当前工作目录
os.chdir(Folder_Path)
# 将该文件夹下的所有文件名存入一个列表
file_list = os.listdir()

# 循环遍历列表中各个CSV文件名，并追加到合并后的文件
for i in range(0, len(file_list)):
    f = open(Folder_Path + '\\' + file_list[i], "r")
    reader = csv.reader(f)
    next(reader, None)#跳过header
    row = [row for row in reader]
    out = open(SaveFile_Path + '\\' + SaveFile_Name, 'a+',newline ='')
    write = csv.writer(out, dialect='excel')#write就像是一个盒子，用来两头传递数据
    write.writerows(row)
