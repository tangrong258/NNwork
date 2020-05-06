import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import statsmodels.formula.api as smf
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.api import qqplot
from datetime import *
import xlsxwriter

from openpyxl import load_workbook
# time_step = 40

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
    data = np.c_[range(1, time_step), r[1:], q, p]#np.hstack, 对一个一维的向量操作，会默认是列向量, range 最多size（40）
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    # print(table.set_index('lag'))
    start = 1900 + len(timesq)
    predict_dta = arma_mod80.predict(str(start), str(start), dynamic=True)
    # print(predict_dta)
    pred = np.array(predict_dta.array)
    # a = np.square(testy - pred)
    b = np.abs(testy - pred)
    # c = np.abs(testy - pred) / testy
    return pred[0], np.mean(b)

if __name__ == '__main__':

    MAEscalaro = np.zeros((3, 10))
    MAEscalaru = np.zeros((3, 10))
    for t in range(5, 6):
        time_step = t*5 + 5

        with open(r"D:\轨迹预测\prediction\ARIMA\test_xo.pkl", 'rb') as f:
            test_xo6 = pickle.load(f)
        with open(r"D:\轨迹预测\prediction\ARIMA\test_yo.pkl", 'rb') as f:
            test_yo6 = pickle.load(f)

        sto = len(test_yo6)-100
        MAEo = np.zeros((3, 100))
        predo = np.zeros((3, 100))
        for k in range(0, 3):
            test_xo = np.zeros((test_xo6.shape[0], time_step), dtype=np.float32)
            for i in range(test_xo.shape[0]):
                for j in range(time_step):
                    test_xo[i, j] = test_xo6[i, j, k]

            test_yo = test_yo6[:,  k]#预测值超过两个就是三维的

            for i in range(sto, len(test_yo)):
                testxo = test_xo[i]
                testyo = test_yo[i]
                predo[k, i-sto], MAEo[k, i-sto] = ARIMA(testxo, testyo, time_step)
            MAEscalaro[k, t] = np.mean(MAEo[k, :])




        with open(r"D:\轨迹预测\prediction\ARIMA\test_xu.pkl", 'rb') as f:
            test_xu6 = pickle.load(f)
        with open(r"D:\轨迹预测\prediction\ARIMA\test_yu.pkl", 'rb') as f:
            test_yu6 = pickle.load(f)

        stu = len(test_yu6)-65
        MAEu = np.zeros((3, 65))
        predu = np.zeros((3, 65))
        for k in range(0, 3):
            test_xu = np.zeros((test_xu6.shape[0], time_step), dtype=np.float32)
            for i in range(test_xu.shape[0]):
                for j in range(time_step):
                    test_xu[i, j] = test_xu6[i, j, k]

            test_yu = test_yu6[:,  k]#预测值超过两个就是三维的

            for i in range(stu, len(test_yu)):
                testxu = test_xu[i]
                testyu = test_yu[i]
                predu[k, i-stu], MAEu[k, i-stu] = ARIMA(testxu, testyu, time_step)
            MAEscalaru[k, t] = np.mean(MAEu[k, :])


    dt = pd.DataFrame(MAEscalaro)
    dt.to_csv(r"D:\轨迹预测\prediction\ARIMA\MAEo.csv")

    dt = pd.DataFrame(predo)
    dt.to_csv(r"D:\轨迹预测\prediction\ARIMA\predo.csv")

    dt = pd.DataFrame(MAEscalaru)
    dt.to_csv(r"D:\轨迹预测\prediction\ARIMA\MAEu.csv")

    dt = pd.DataFrame(predu)
    dt.to_csv(r"D:\轨迹预测\prediction\ARIMA\predu.csv")
