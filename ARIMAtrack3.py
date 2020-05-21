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

    MAEscalar = np.zeros((3, 10))
    for t in range(5, 6):
        time_step = t*5 + 5

        root = r"D:\轨迹预测\prediction\ARIMA\nonscalar"
        with open(root + '\\' + 'test_x' + str(time_step) + ".pkl", 'rb') as f:
            test_x6 = pickle.load(f)
        with open(root + '\\' + 'test_y' + str(time_step) + ".pkl", 'rb') as f:
            test_y6 = pickle.load(f)

        st = len(test_y6)-100
        MAE = np.zeros((3, 100))
        pred = np.zeros((3, 100))
        for k in range(0, 3):
            test_x = np.zeros((test_x6.shape[0], time_step), dtype=np.float32)
            for i in range(test_x6.shape[0]):
                for j in range(time_step):
                    test_x[i, j] = test_x6[i, j, k]

            # test_y = np.zeros((test_y6.shape[0], 1), dtype = np.float32)
            # for i in range(test_y6.shape[0]):
            #     test_y[i, 0] = test_y6[i, 0]
            test_y = test_y6[:,  k]#预测值超过两个就是三维的


            # error = np.zeros((45, 3))
            # st = len(test_y)-50

            for i in range(st, len(test_y)):
                testx = test_x[i]
                testy = test_y[i]
                # predhxy[i-st, k],
                pred[k, i-st], MAE[k, i-st] = ARIMA(testx, testy, time_step)
            MAEscalar[k, t] = np.mean(MAE[k, :])
        # print("time_step:", time_step)
        # dt = pd.DataFrame(predhxy[15:50, :])

        # dt.to_csv(r"D:\轨迹预测\prediction"+"\\" + "ARIMApred.csv")

        # print('MAE:', MAE)

    dt = pd.DataFrame(MAEscalar)
    dt.to_csv(r"D:\轨迹预测\prediction\ARIMA\MAE1.csv")

    dt = pd.DataFrame(pred)
    dt.to_csv(r"D:\轨迹预测\prediction\ARIMA\pred1.csv")
