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
    data = np.c_[range(1, 41), r[1:], q, p]#np.hstack, 对一个一维的向量操作，会默认是列向量, range 最多size（40）
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

if __name__ == '__main__':

    MAEscalar = np.zeros((3, 10))
    for t in range(8, 10):
        time_step = t*5 + 5
        root = r"D:\轨迹预测\prediction\ARIMA\scalar"
        with open(root + '\\' + 'test_x' + str(time_step) + ".pkl", 'rb') as f:
            test_x6 = pickle.load(f)
        with open(root + '\\' + 'test_y' + str(time_step) + ".pkl", 'rb') as f:
            test_y6 = pickle.load(f)

        predhxy = np.zeros((50, 3))
        MAE = np.zeros((50, 3))
        for k in range(0, 3):
            test_x = np.zeros((50, time_step), dtype=np.float32)
            test_y = np.zeros(50, dtype=np.float32)
            MAE_batch = np.zeros((10, 3), dtype=np.float32)
            for s in range(0, 10):
                for i in range(0, 50):
                    index = i+50*s+2500
                    print('index:', index)
                    test_y[i] = test_y6[index, k]
                    for j in range(time_step):
                        test_x[i, j] = test_x6[index, j, k]

                for i in range(0, len(test_y)):
                    testx = test_x[i]
                    testy = test_y[i]
                    predhxy[i, k], MAE[i, k] = ARIMA(testx, testy, time_step)
                    MAE_batch[s, k] = np.mean(MAE[:, k])
            MAEscalar[k, t] = np.mean(MAE_batch[:, k])
        print("time_step:", time_step)

    dt = pd.DataFrame(MAEscalar)
    dt.to_csv(r"D:\轨迹预测\prediction\ARIMA" + "\\" + "MAE10_50.csv")

