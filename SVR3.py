from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.formula.api as smf
from scipy import stats
import statsmodels.api as sm
import numpy as np
import pickle
import pandas as pd

from openpyxl import load_workbook


def SVRR(x, y):
    x_test = x[0:int(len(x) * 0.5), :]
    y_test = y[0:int(len(y) * 0.5)]

    x_train = x[int(len(x) * 0.5): len(x), :]
    y_train = y[int(len(y) * 0.5): len(y)]


    # 2 分割训练数据和测试数据
    # 随机采样25%作为测试 75%作为训练
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=33)
    # 最终和别的方法进行归集对比，不能随机求取测试集

    # 3 训练数据和测试数据进行标准化处理
    # ss_x = StandardScaler()
    # ss_x = MinMaxScaler()
    # x_train = ss_x.fit_transform(x_train)
    # x_test = ss_x.transform(x_test)
    #
    # # ss_y = StandardScaler()
    # ss_y = MinMaxScaler()
    # y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
    # y_test = ss_y.transform(y_test.reshape(-1, 1))

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

MAEscalar = np.zeros((3, 10))



# with open(r"D:\轨迹预测\prediction\SVR\test_x.pkl", 'rb') as f:
#     test_x = pickle.load(f)
# with open(r"D:\轨迹预测\prediction\SVR\test_y.pkl", 'rb') as f:
#     test_y = pickle.load(f)
# with open(r"D:\轨迹预测\prediction\SVR\test_y12.pkl", 'rb') as f:
#     test_y2 = pickle.load(f)
# with open(r"D:\轨迹预测\prediction\SVR\test_y123.pkl", 'rb') as f:
#     test_y3 = pickle.load(f)

with open(r"D:\轨迹预测\prediction\ARIMA\test_x.pkl", 'rb') as f:
    test_x = pickle.load(f)
with open(r"D:\轨迹预测\prediction\ARIMA\test_y.pkl", 'rb') as f:
    test_y = pickle.load(f)
with open(r"D:\轨迹预测\prediction\ARIMA\test_y12.pkl", 'rb') as f:
    test_y2 = pickle.load(f)
with open(r"D:\轨迹预测\prediction\ARIMA\test_y123.pkl", 'rb') as f:
    test_y3 = pickle.load(f)

def ARIMA(testx, testy,time_step):
    timesq = pd.date_range(start="2018/1/01 00:00", periods = time_step, freq='30min')
    dta = pd.Series(testx, timesq)
    dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1900', length=len(timesq)))
    # dta = dta.diff().dropna()
    arma_mod80 = sm.tsa.ARIMA(dta, (1, 0, 0)).fit()
    # print(arma_mod80.aic, arma_mod80.bic, arma_mod80.hqic)
    resid = arma_mod80.resid
    # print(sm.stats.durbin_watson(arma_mod80.resid.values))
    stats.normaltest(resid)
    r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    if time_step <= 40:
        u = time_step
    else:
        u = 41
    data = np.c_[range(1, u), r[1:], q, p]#np.hstack, 对一个一维的向量操作，会默认是列向量, range 最多size（40）
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    # print(table.set_index('lag'))
    start = 1900 + len(timesq)
    predict_dta = arma_mod80.predict(str(start), str(start), dynamic=True)
    # print(predict_dta)
    pred = np.array(predict_dta.array)
    # a = np.square(testy - pred)
    b = np.abs(testy - pred)
    # c = np.abs(testy - pred) / testy
    return pred, np.mean(b)


"""
pred1 = np.zeros((3000, 3))
pred2 = np.zeros((1500, 3))
pred3 = np.zeros((750, 3))
for k in range(0, 3):
    x = test_x[len(test_x) - 7000:len(test_x)-1000, :, k]
    y = test_y[len(test_x) - 7000:len(test_x)-1000, k]
    pred1[:, k], MAEscalar[k, 1] = SVRR(x, y)

    x2 = np.hstack((x[3000:6000, :], pred1[:, :]))
    y2 = test_y2[len(test_x) - 4000:len(test_x)-1000, -1, k]
    pred2[:, k], MAEscalar[k, 2] = SVRR(x2, y2)

    x3 = np.hstack((x2[1500:3000, :], pred2[:, :]))
    y3 = test_y3[len(test_x) - 2500:len(test_x)-1000, -1, k]
    pred3[:, k], MAEscalar[k, 3] = SVRR(x3, y3)



    dt = pd.DataFrame(pred1)
    dt.to_csv(r"D:\轨迹预测\prediction\SVR\pred1.csv")
    dt = pd.DataFrame(MAEscalar)
    dt.to_csv(r"D:\轨迹预测\prediction\SVR\MAE123.csv")
"""
pred1 = np.zeros((1000, 3))
mae1 = pred1
pred2 = np.zeros((1000, 3))
mae2 = pred2
pred3 = np.zeros((1000, 3))
mae3 = pred3

low = 2000
up = 1000
for k in range(0, 3):
    x = test_x[len(test_x) - low:len(test_x)-up, :, k]
    y = test_y[len(test_x) - low:len(test_x)-up, k]
    for i in range(0, len(x)):
        pred1[i, k], mae1[i, k] = ARIMA(x[i], y[i], len(x[i]))
    MAEscalar[k, 1] = np.mean(mae1[:, k])

    x2 = np.hstack((x, np.reshape(pred1[:, k], [pred1.shape[0], 1])))
    y2 = test_y2[len(test_x) - low:len(test_x)-up, -1, k]

    for i in range(0, len(x2)):
        pred2[i, k], mae2[i, k] = ARIMA(x2[i], y2[i], len(x2[i]))
    MAEscalar[k, 2] = np.mean(mae2[:, k])

    x3 = np.hstack((x2, np.reshape(pred2[:, k], [pred2.shape[0], 1])))
    y3 = test_y3[len(test_x) - low:len(test_x)-up, -1, k]
    for i in range(0, len(x3)):
        pred3[i, k], mae3[i, k] = ARIMA(x3[i], y3[i], len(x3[i]))
    MAEscalar[k, 3] = np.mean(mae3[:, k])

    dt = pd.DataFrame(pred1)
    dt.to_csv(r"D:\轨迹预测\prediction\ARIMA\pred1.csv")
    dt = pd.DataFrame(MAEscalar)
    dt.to_csv(r"D:\轨迹预测\prediction\ARIMA\MAE123.csv")


