from mylib import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

#1. Get and modify data
msft = pd.read_csv("data/msft.csv")
aapl = pd.read_csv("data/aapl.csv")
class SModel_JD:
	def __init__(self, stock, start, end):
		
		self.start = start
		self.end = end
		stock = stock[(stock["date"] > start)
						& (stock["date"] < end)]

		stock["date"] = pd.to_datetime(stock["date"])
		cols = ["DATE", "STOCK"]
		stock.columns = cols
		self.stock = stock

	def create_market_env(self,name):
		self.pricing_date = dt.datetime.strptime(self.start, "%Y-%m-%d")
		final_date = dt.datetime.strptime(self.end, "%Y-%m-%d")

		me_stock = market_env(name, self.pricing_date)
		initial_value = data.iloc[0]["STOCK"]
		me_stock.add_constant("initial_value", initial_value)
		me_stock.add_constant("final_date", final_date)
		me_stock.add_constant("currency", "USD")
		me_stock.add_constant("frequency", "B")
		me_stock.add_constant("paths", 10000)
		csr = constant_short_rate("csr", 0.01)
		me_stock.add_curve("discount_curve", csr)
		me_stock.add_constant("lambda", 15.0)
		me_stock.add_constant("mu", 1.0)
		me_stock.add_constant("delta", 0.1)

		self.stocks_model = jump_diffusion("stocks_model", me_stock)
		self.me_stock = me_stock

	def calc_model_values(self,p0):
		i = 0
		j = 0
		lamb, mu, delt, vol = p0
		stocks_model.update(lamb = lamb, mu = mu, delt = delt, volatility = vol)
		model_values = {}
		for s in self.stock["close"]:
			if i % 10 == 0:
				model_values["stock"+str(j)] = self.stock["close"][i]
				j+=1
		self.model_values = model_values
	def mean_squared_error(self,p0):
		pass

	def strategy(self):
		pass














