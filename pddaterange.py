import pandas as pd
timesp = pd.date_range(start="2018/1/01 00:00",end = "2018/5/31 23:00",freq="30min")
print(timesp)
print(timesp[3],len(timesp))