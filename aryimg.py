import  pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as pyplot
import imageio
import math
import os
with open(r"D:\ZSNJAP01\flight\delaydata\linedlyt.pkl",'rb') as f:
    info = pickle.load(f)

infoint = np.zeros(((len(info),len(info[0]),len(info[0,0]))),dtype= int)#[n,4,5]
for k in range(len(info)):
    for i in range(len(info[k])):
        for j in range(len(info[k,i])):
            infoint[k,i,j] = int(math.ceil(info[k,i,j]))

def imadjust(imgdata, low_in, high_in, low_out, high_out):
    h, w = imgdata.shape#
    f1 = np.zeros((h, w),dtype= int)
    for i in range(0, h):
        for j in range(0, w):
                f1[i, j] = ((imgdata[i, j]-low_in)/(high_in - low_in))*(high_out - low_out)
    return f1

def out_img(data,k):                 #输出图片
    new_im = Image.fromarray(data,'L')     #调用Image库,数组转换为image
    #new_im.show()               #显示新图片
    imageio.imwrite(r'D:\ZSNJAP01\flight\imagetime' + '\\' + 'img_'+str(k)+'.jpg', new_im)   #保存图片到本地

b = os.path.exists(r'D:\ZSNJAP01\flight\imagetime')
if b:
    print("path exist")
else:
    os.makedirs(r'D:\ZSNJAP01\flight\imagetime')

for k in range(len(infoint)):
    # info255 = imadjust(infoint[k],np.min(infoint.min(0)), np.max(infoint.max(0)), 0, 255)#先不用
    print(info[k])
    out_img(info[k],k)