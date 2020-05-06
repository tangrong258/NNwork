import pickle
import pandas  as pd
import numpy as np
with open(r"D:\ZSNJAP01\flight\delaydata\linez.pkl",'rb') as f:
    featime = pickle.load(f) #ndarray dim = [3623,4,5]
featime = np.reshape(featime,[-1,20])

df = pd.DataFrame(featime )
df = df
df.to_csv(r"D:\ZSNJAP01\flight\delaydata\linez.csv", index=False, sep=',')