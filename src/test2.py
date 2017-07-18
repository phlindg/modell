from mylib import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model_stock import SModel

#dx-analytics.com!!!!!!
#https://gist.github.com/yhilpisch/648565d3d5d70663b7dc418db1b81676


msft = pd.read_csv("data/msft.csv")
aapl = pd.read_csv("data/aapl.csv")
atvi = pd.read_csv("data/atvi.csv")
gm = pd.read_csv("data/gm.csv")

start = "2014-01-01"
end = "2017-06-01" # MASTE FIIXA DATUMEN
model_start = start
model_end = end #Ju mer tid det gar ju mer lognormal blir den.
stock = SModel(gm, start, end,model_start,model_end,1000, "GBM")
returns = stock.calc_returns().dropna()
vol_ret = returns.std() * np.sqrt(252.)
print("VOL: ", vol_ret)
if stock.model_choice == "JD":
	stock.create_market_env("me_stocks", 15.0, .1, 0.1, vol_ret)
if stock.model_choice == "GBM":
	stock.create_market_env("me_stocks", vol_ret)

ranges_jd = ((5.0, 15.0, 1.0),
			(-0.5, 0.5,  0.1),
			(0.0, 0.05, 0.01),
			(0.0, 2.0, 0.2))
ranges_gbm = (0.0, 25.0, 1.0)
already_known = (24.181197131, 0.0534761091876, 0.127646433484, 0.0120390604717)
hmm = (9.8530443006, -0.00443875543192, 0.00366927778959, vol_ret)
if stock.model_choice == "JD":
	stock.fit(ranges_jd, hmm)
if stock.model_choice == "GBM":
	stock.fit(ranges_gbm,5.0)

stock.bollinger(0.05, 2)

