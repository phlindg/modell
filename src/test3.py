
from mylib import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl
quandl.ApiConfig.api_key = 'ZazfayyxoPXJCJjhBi2_'
from model_stock import SModel

from strategies import *

def get_stocks(tickers, start,end):
	stocks= []
	for ticker in tickers:
		stock = quandl.get("WIKI/"+ticker,
				start_date = start,
				end_date = end)
		stocks.append(stock)
		print(stock)

	mydf = pd.DataFrame(stocks)
	#print(mydf)
def get_stock(ticker):
	stock = quandl.get("WIKI/"+ticker)

	stock.to_csv("data/"+ticker+".csv")


start = "2016-01-01"
end = "2017-01-01"
model_end = "2017-02-01"

#get_stock("ATVI")
gm = pd.read_csv("data/GM.csv")
f = pd.read_csv("data/F.csv")
atvi = pd.read_csv("data/ATVI.csv")
stocks = [atvi]
stock_names = ["F"]

strat = Strategy(stocks, stock_names,start,end)
#signals = strat.choose("pairs_trade", 30, "MEAN")
signals = strat.choose("mean_rev", 30, 5)
strat.set_signals(signals)
strat.gen_trades()


strat.learn(end)
#strat.plot_strat()
#strat.plotem()