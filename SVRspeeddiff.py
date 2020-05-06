from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pickle
import pandas as pd
import  xlsxwriter
from openpyxl import load_workbook


def SVRR(x, y):
    x_train = x[0:int(len(x) * 0.75), :]
    y_train = y[0:int(len(y) * 0.75)]

    x_test = x[int(len(x) * 0.75): len(x), :]
    y_test = y[int(len(y) * 0.75): len(y)]

    # 2 分割训练数据和测试数据
    # 随机采样25%作为测试 75%作为训练
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
    # 最终和别的方法进行归集对比，不能随机求取测试集



    # 4.1 支持向量机模型进行学习和预测
    # 线性核函数配置支持向量机
    linear_svr = SVR(kernel="linear")
    # 训练
    linear_svr.fit(x_train, y_train)
    # 预测 保存预测结果
    linear_svr_y_predict = linear_svr.predict(x_test)
    # linear_svr_y_predict = linear_svr_y_predict.reshape(-1, 1)
    # pred = ss_y.inverse_transform(linear_svr_y_predict[:])

    # 多项式核函数配置支持向量机
    poly_svr = SVR(kernel="poly")
    # 训练
    poly_svr.fit(x_train, y_train)
    # 预测 保存预测结果
    poly_svr_y_predict = poly_svr.predict(x_test)
    # poly_svr_y_predict = poly_svr_y_predict.reshape(-1, 1)

    # 高斯核函数
    rbf_svr = SVR(kernel="rbf")
    # 训练
    rbf_svr.fit(x_train, y_train)
    # 预测 保存预测结果
    rbf_svr_y_predict = rbf_svr.predict(x_test)
    # rbf_svr_y_predict = rbf_svr_y_predict.reshape(-1, 1)


    #高斯
    pred = rbf_svr_y_predict[:]

    # 5 模型评估
    # 线性核函数 模型评估
    print("线性核函数支持向量机的默认评估值为：", linear_svr.score(x_test, y_test))
    print("线性核函数支持向量机的R_squared值为：", r2_score(y_test, linear_svr_y_predict))
    print("线性核函数支持向量机的均方误差为:", mean_squared_error(y_test,
                                                  linear_svr_y_predict))
    print("线性核函数支持向量机的平均绝对误差为:", mean_absolute_error(y_test,
                                                     linear_svr_y_predict))
    # 对多项式核函数模型评估
    print("对多项式核函数的默认评估值为：", poly_svr.score(x_test, y_test))
    print("对多项式核函数的R_squared值为：", r2_score(y_test, poly_svr_y_predict))
    print("对多项式核函数的均方误差为:", mean_squared_error(y_test,
                                               poly_svr_y_predict))
    print("对多项式核函数的平均绝对误差为:", mean_absolute_error(y_test,
                                                  poly_svr_y_predict))

    # 高斯核函数评估
    print("对高斯核函数的默认评估值为：", rbf_svr.score(x_test, y_test))
    print("对高斯核函数的R_squared值为：", r2_score(y_test, rbf_svr_y_predict))
    print("对高斯核函数的均方误差为:", mean_squared_error(y_test,
                                              rbf_svr_y_predict))
    print("对高斯核函数的平均绝对误差为:", mean_absolute_error(y_test,
                                                 rbf_svr_y_predict))

    MAE = mean_absolute_error(y_test, rbf_svr_y_predict)
    return pred, MAE

MAEscalar1 = np.zeros((3, 10))
MAEscalar2 = np.zeros((3, 10))
for t in range(0, 1):
    time_step = t*5 + 5
    with open(r"D:\轨迹预测\prediction\SVR\test_xo.pkl", 'rb') as f:
        test_x = pickle.load(f)
    with open(r"D:\轨迹预测\prediction\SVR\test_yo.pkl", 'rb') as f:
        test_y = pickle.load(f)
    with open(r"D:\轨迹预测\prediction\SVR\test_xu.pkl", 'rb') as f:
        test_x2 = pickle.load(f)
    with open(r"D:\轨迹预测\prediction\SVR\test_yu.pkl", 'rb') as f:
        test_y2 = pickle.load(f)



    pred1 = np.zeros((1000, 3))
    pred2 = np.zeros((588, 3))

    for k in range(0, 3):
        x = test_x[len(test_x) - 4000:len(test_x), :, k]
        y = test_y[len(test_x) - 4000:len(test_x), k]
        pred1[:, k], MAEscalar1[k, t] = SVRR(x, y)

        x2 = test_x2[len(test_x2) - 2352:len(test_x2), :, k]
        y2 = test_y2[len(test_x2) - 2352:len(test_x2), k]
        pred2[:, k], MAEscalar2[k, t] = SVRR(x2, y2)


    dt = pd.DataFrame(pred1)
    dt.to_csv(r"D:\轨迹预测\prediction\SVR\predo.csv")
    dt = pd.DataFrame(MAEscalar1)
    dt.to_csv(r"D:\轨迹预测\prediction\SVR\MAEo.csv")

    dt = pd.DataFrame(pred2)
    dt.to_csv(r"D:\轨迹预测\prediction\SVR\predu.csv")
    dt = pd.DataFrame(MAEscalar2)
    dt.to_csv(r"D:\轨迹预测\prediction\SVR\MAEu.csv")



