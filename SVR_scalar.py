from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pickle
import pandas as pd
import xlsxwriter
from openpyxl import load_workbook
import pandas as pd



MAEscalar = np.zeros((3, 10))
for t in range(0, 10):
    time_step = t*5 + 5
    root = r"D:\轨迹预测\prediction\SVR\scalar"
    with open(root + '\\' + 'test_x' + str(time_step) + ".pkl", 'rb') as f:
        test_x = pickle.load(f)
    with open(root + '\\' + 'test_y' + str(time_step) + ".pkl", 'rb') as f:
        test_y = pickle.load(f)
    with open(root + '\\' + 'train_x' + str(time_step) + ".pkl", 'rb') as f:
        train_x = pickle.load(f)
    with open(root + '\\' + 'train_y' + str(time_step) + ".pkl", 'rb') as f:
        train_y = pickle.load(f)

    # if train_x.shape[0] > 2000:
    #     train_x = train_x[0:2000, :, :]
    #     train_y = train_y[0:2000, :]
    #
    # if test_x.shape[0] > 500:
    #     test_x = test_x[0:500, :, :]
    #     test_y = test_y[0:500, :]

    x_test = test_x[0:1000, :, :]
    y_test = test_y[0:1000, :]

    x_train = train_x[1000:2000, :, :]
    y_train = train_y[1000:2000, :]

    pred = np.zeros((len(y_test), 3))
    for k in range(0, 3):
        x_test = test_x[0:1000, :, k]
        y_test = test_y[0:1000, k]

        x_train = train_x[1000:2000, :, k]
        y_train = train_y[1000:2000, k]


        # 4.1 支持向量机模型进行学习和预测
        # 线性核函数配置支持向量机
        linear_svr = SVR(kernel="linear")
        # 训练
        linear_svr.fit(x_train, y_train)
        # 预测 保存预测结果
        linear_svr_y_predict = linear_svr.predict(x_test)


        # 多项式核函数配置支持向量机
        poly_svr = SVR(kernel="poly")
        # 训练
        poly_svr.fit(x_train, y_train)
        # 预测 保存预测结果
        poly_svr_y_predict = poly_svr.predict(x_test)



        # 高斯核函数
        rbf_svr = SVR(kernel="rbf")
        # 训练
        rbf_svr.fit(x_train, y_train)
        # 预测 保存预测结果
        rbf_svr_y_predict = rbf_svr.predict(x_test)


        #选择高斯
        pred[:, k] = rbf_svr_y_predict[:]



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

        MAEscalar[k, t] = mean_absolute_error(y_test, rbf_svr_y_predict)


dt = pd.DataFrame(pred)
dt.to_csv(r"D:\轨迹预测\prediction\SVR\pred_y.csv")

dt = pd.DataFrame(MAEscalar)
dt.to_csv(r"D:\轨迹预测\prediction\SVR\MAE5_50.csv")

