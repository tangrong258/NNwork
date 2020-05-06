import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import statsmodels.formula.api as smf
from scipy import stats
import statsmodels.api as sm
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


timesq = pd.date_range(start="2018/1/01 00:00", periods = 48, freq = howlong)
sig = flytime[:, 0]
train = sig[0:len(timesq)]
test = sig[len(timesq):]


dta = pd.Series(train, timesq)
# train_start = pd.to_datetime('2018-01-01',format = '%Y-%m-%d')
dta.index = pd.Index(sm.tsa.datetools.dates_from_range( '2000', length = 48))
dta.plot(figsize=(12, 8))
# plt.show()


# fig = plt.figure(figsize=(12,8))
# ax1= fig.add_subplot(111)
diff1 = dta.diff(1).dropna()
# diff1.plot(ax=ax1)

# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)
# plt.show()

# arma_mod50 = sm.tsa.ARMA(dta,(5,0)).fit()
# print(arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)
# arma_mod30 = sm.tsa.ARMA(dta,(3,0)).fit()
# print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
# arma_mod71 = sm.tsa.ARMA(dta,(7,1)).fit()
# print(arma_mod71.aic,arma_mod71.bic,arma_mod71.hqic)
arma_mod80 = sm.tsa.ARMA(dta, (8,0)).fit()
print(arma_mod80.aic,arma_mod80.bic,arma_mod80.hqic)

resid = arma_mod80.resid
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
# plt.show()

print(sm.stats.durbin_watson(arma_mod80.resid.values))

stats.normaltest(resid)

# fig = plt.figure(figsize=(12,8))
# ax = fig.add_subplot(111)
# fig = qqplot(resid, line='q', ax=ax, fit=True)
# plt.show()

r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

predict_dta = arma_mod80.predict('2040', '2055', dynamic=True)
print(np.array(predict_dta.array))


# fig, ax = plt.subplots(figsize=(12, 8))
# ax = dta.ix['2000':].plot(ax=ax)
# fig = arma_mod80.plot_predict('2048', '2055', dynamic=True, ax=ax, plot_insample=False)
# plt.show()
