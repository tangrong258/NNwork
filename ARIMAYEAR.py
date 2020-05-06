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
howlong = "30min"
sp = 30
rootfrom = r"D:\ZSNJAP01\flight\delaydata" + "\\"+ howlong
with open(rootfrom + "\\" + "lineavgt.pkl",'rb') as f:
    featime = pickle.load(f)

featime2d = np.reshape(featime,[-1,featime.shape[1] * featime.shape[2]])
flytime = np.zeros((featime2d.shape[0], 6), dtype='float32')
flytime[:, 0] = featime2d[:, 2]
flytime[:, 1] = featime2d[:, 7]
flytime[:, 2] = featime2d[:, 12]
flytime[:, 3] = featime2d[:, 15]
flytime[:, 4] = featime2d[:, 16]
flytime[:, 5] = featime2d[:, 18]

for i in range (flytime.shape[1]):
    for j in range(flytime.shape[0]):
        if flytime[j,i] >= 10.0:
            flytime[j,i] = np.mean(flytime[:,i])
        if flytime[j,i] <= 0.0:
            flytime[j, i] = 0.01

scaler1= MinMaxScaler(feature_range=(0, 1))
flytime = scaler1.fit_transform(flytime).squeeze()

m = flytime.shape[0]
time_step = 48
train_end = int((31+28+31)*24*(60/sp)) - time_step
test_end = int(m) - time_step
test_step = test_end - train_end

def generate_data1(seq):
    x = []
    y = []
    for i in range(len(seq) - time_step):
        x.append([seq[i:i + time_step]])
        y.append([seq[i + time_step]])
    xs = np.array(x, dtype = np.float32).squeeze()
    ys = np.array(y, dtype = np.float32).squeeze()
    return xs, ys

def ARIMA(testx, testy,time_step):
    timesq = pd.date_range(start="2018/1/01 00:00", periods = time_step, freq=howlong)
    dta = pd.Series(testx, timesq)
    dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2000', length=len(timesq)))
    arma_mod80 = sm.tsa.ARMA(dta, (8,0)).fit()
    # print(arma_mod80.aic, arma_mod80.bic, arma_mod80.hqic)
    resid = arma_mod80.resid
    # print(sm.stats.durbin_watson(arma_mod80.resid.values))
    stats.normaltest(resid)
    r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    data = np.c_[range(1, 41), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    # print(table.set_index('lag'))
    start = 2000 + len(timesq)
    predict_dta = arma_mod80.predict(str(start), str(start), dynamic=True)
    # print(predict_dta)
    pred = np.array(predict_dta.array)
    a = np.square(testy - pred)
    b = np.abs(testy - pred)
    c = np.abs(testy - pred) / testy
    return a,b,c

if __name__ == '__main__':
    root = r"D:\ZSNJAP01\flight\prediction\ARIMA" + "\\" + howlong
    sub = "lineavgt"
    x6, y6 = generate_data1(flytime)
    test_x6 = x6[train_end:test_end]
    test_y6 = y6[train_end:test_end]

    test_x = np.zeros((test_x6.shape[0],time_step),dtype = np.float32)
    for i in range(test_x6.shape[0]):
        for j in range(time_step):
            test_x[i, j] = test_x6[i, j, 0]

    test_y = np.zeros((test_y6.shape[0],1),dtype = np.float32)
    for i in range(test_y6.shape[0]):
        test_y[i, 0] = test_y6[i, 0]

    error = np.zeros((100, 3))
    for i in range(100):
        testx = test_x[i]
        testy = test_y[i]
        error[i, :] = ARIMA(testx, testy, time_step)
        print("test_step:", i)

    b = os.path.exists(root + "\\" + sub)
    if b:
        print("path exist")
    else:
        os.makedirs(root + "\\" + sub )
    dt = pd.DataFrame(error)
    dt.to_csv(root + "\\" + sub + "\\" + "error.csv")