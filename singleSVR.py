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




time_step = 6
howlong = '10min'
root = r'D:\ZSNJAP01\flight\prediction\SVR'+'\\'+howlong
index = [0, 2, 3, 4]
for d in index:
    with open(root + '\\' + 'test_x.pkl', 'rb') as f:
        test_x = pickle.load(f)
    with open(root + '\\' + 'test_y' + str(d) + '.pkl', 'rb') as f:
        test_y = pickle.load(f)
    with open(root + '\\' + 'train_x.pkl', 'rb') as f:
        train_x = pickle.load(f)
    with open(root + '\\' + 'train_y' + str(d) + '.pkl', 'rb') as f:
        train_y = pickle.load(f)

    error = np.zeros((len(test_y), 3))
    pred = np.zeros((len(test_y), 1))

    x_test = np.reshape(test_x, [-1, test_x.shape[1]*test_x.shape[2]])

    y_test = test_y

    x_train = np.reshape(train_x, [-1, train_x.shape[1]*train_x.shape[2]])
    y_train = train_y

    # 3 训练数据和测试数据进行标准化处理
    # ss_x = StandardScaler()
    ss_x = MinMaxScaler()
    x_train = ss_x.fit_transform(x_train)
    x_test = ss_x.transform(x_test)

    # ss_y = StandardScaler()
    ss_y = MinMaxScaler()
    y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
    y_test = ss_y.transform(y_test.reshape(-1, 1))




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
    pred[:, 0] = rbf_svr_y_predict[:]



    # 5 模型评估
    # # 线性核函数 模型评估
    # print("线性核函数支持向量机的默认评估值为：", linear_svr.score(x_test, y_test))
    # print("线性核函数支持向量机的R_squared值为：", r2_score(y_test, linear_svr_y_predict))
    # print("线性核函数支持向量机的均方误差为:", mean_squared_error(y_test,
    #                                               linear_svr_y_predict))
    # print("线性核函数支持向量机的平均绝对误差为:", mean_absolute_error(y_test,
    #                                                 linear_svr_y_predict))
    # # 对多项式核函数模型评估
    # print("对多项式核函数的默认评估值为：", poly_svr.score(x_test, y_test))
    # print("对多项式核函数的R_squared值为：", r2_score(y_test, poly_svr_y_predict))
    # print("对多项式核函数的均方误差为:", mean_squared_error(y_test,
    #                                            poly_svr_y_predict))
    # print("对多项式核函数的平均绝对误差为:", mean_absolute_error(y_test,
    #                                               poly_svr_y_predict))
    #
    # # 高斯核函数评估
    # print("对高斯核函数的默认评估值为：", rbf_svr.score(x_test, y_test))
    # print("对高斯核函数的R_squared值为：", r2_score(y_test, rbf_svr_y_predict))
    # print("对高斯核函数的均方误差为:", mean_squared_error(y_test,
    #                                            rbf_svr_y_predict))
    # print("对高斯核函数的平均绝对误差为:", mean_absolute_error(y_test,
    #                                             rbf_svr_y_predict))

    # y_test = ss_y.inverse_transform(y_test)
    # pred = ss_y.inverse_transform(pred)
    for i in range(0, len(y_test)):
        error[i, 0] = np.square(y_test[i] - pred[i])
        error[i, 1] = np.abs(y_test[i] - pred[i])
        error[i, 2] = np.abs(y_test[i] - pred[i])/y_test[i]

    dt = pd.DataFrame(error)
    dt.to_csv(root + "\\" + "\\" + str(d) + 'error' + str(time_step) + ".csv")

