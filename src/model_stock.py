from mylib import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import scipy.optimize as spo
import random

#1. Get and modify data
msft = pd.read_csv("data/msft.csv")
aapl = pd.read_csv("data/aapl.csv")
i=0
class SModel:
	def __init__(self, stock, start, end,model_end, model_choice):
		
		self.start = start
		self.end = end
		self.model_end = model_end
		self.model_choice = model_choice
		stock = stock[(stock["date"] > start)
						& (stock["date"] < end)]

		stock.loc[:,"date"] = pd.to_datetime(stock.loc[:,"date"])
		cols = ["DATE", "STOCK"]
		stock.columns = cols
		self.stock = stock

	def create_market_env(self,name, vol,lamb=None, mu=None, delt=None):
		self.pricing_date = dt.datetime.strptime(self.end, "%Y-%m-%d")
		final_date = dt.datetime.strptime(self.model_end, "%Y-%m-%d")
		paths = 10000
		me_stock = market_env(name, self.pricing_date)
		initial_value = self.stock.iloc[-1]["STOCK"]

		me_stock = market_env(name, self.pricing_date)
		me_stock.add_constant("initial_value", initial_value)
		me_stock.add_constant("final_date", final_date)
		me_stock.add_constant("currency", "USD")
		me_stock.add_constant("frequency", "B")
		me_stock.add_constant("paths", paths)
		csr = constant_short_rate("csr", 0.01)
		me_stock.add_curve("discount_curve", csr)
		if self.model_choice == "JD":
			me_stock.add_constant("lambda", lamb)
			me_stock.add_constant("mu", mu)
			me_stock.add_constant("delta", delt)
		me_stock.add_constant("volatility", vol)
		me_stock.add_constant("strike", initial_value)
		me_stock.add_constant("maturity", final_date)
		if self.model_choice == "JD":
			stocks_model = jump_diffusion("stocks_model", me_stock)
		if self.model_choice == "GBM":
			stocks_model = geometric_brownian_motion("stocks_model", me_stock)
		self.stocks_model = stocks_model
		self.me_stock = me_stock

		self.paths = self.stocks_model.get_instrument_values(fixed_seed = True)
		self.stocks_model.generate_time_grid()
		
		
	def calc_model_values(self,p0):
		i = 0
		j = 0
		if self.model_choice == "JD":
			lamb, mu, delt, vol = p0
			self.stocks_model.update(lamb = lamb, mu = mu, delt = delt, volatility = vol)
		if self.model_choice == "GBM":
			vol = p0
			self.stocks_model.update(volatility = vol)
		self.paths = self.stocks_model.get_instrument_values(fixed_seed = True)
		model_values = {}

		for path in self.paths:
			
			model_values["path_"+str(i)] = path
			i+=1
		return model_values
	def mean_squared_error(self,p0):
		global i
		model_values = np.array(list(self.calc_model_values(p0).values()))
		market_values = self.stock["STOCK"].values

		diffs = model_values - market_values
		MSE = np.sum(diffs**2)/len(diffs)
		if self.model_choice == "JD":
			if i % 20 == 0:
				if i == 0:
					print("%4s %6s %6s %6s %6s --> %6s" % ("i","lambda","mu", "delta", "vola", "MSE"))
				print("%4s %6s %6s %6s %6s --> %6s" % (i,p0[0],p0[1],p0[2],p0[3], MSE))
			i+=1
		elif self.model_choice == "GBM":
			if i % 20 == 0:
				if i == 0:
					print("%4s %6s --> %6s" % ("i","vola", "MSE"))
				print("%4s %6s --> %6s" % (i,p0[0], MSE))
			i+=1
		return MSE
	def fit(self, ranges, already_known = None,fit=False):
		if already_known == None:
			opt_global = spo.brute(self.mean_squared_error,
							ranges, finish = None)
			opt_local = spo.fmin(self.mean_squared_error,
						opt_global, xtol = 0.001,
						ftol = 0.0001, maxiter = 1000, maxfun = 1000)
			self.p0 = opt_local
		if fit == True:
			opt_local = spo.fmin(self.mean_squared_error,
						already_known, xtol = 0.001,
						ftol = 0.0001, maxiter = 1000, maxfun = 1000)
			self.p0 = opt_local
		else:
			self.p0 = already_known
		#self.calc_model_values(self.p0)
	def check_prob(self, gain):
		last_val = self.stock["STOCK"].values[-1]
		last_val_model = last_val + gain
		test = []
		
		for path in self.paths:
			print(path[0], last_val_model)
			if path[-1] > last_val_model:
				test.append(1)
			else:
				test.append(0)
		 
		test = np.array(test)
		return np.sum(test)/len(test)
	def check(self):
		hmm = {}
		hmm["MODEL"] = np.array(list(self.calc_model_values(self.p0).values()))
		#hmm["MARKET"] = self.stock["STOCK"].values
		#hmm["ERROR"] = hmm["MODEL"] - hmm["MARKET"]
		#print(hmm["ERROR"])
	def calc_returns(self):
		prices = self.stock["STOCK"]
		return prices.pct_change(1)
	def correla(self):
		k = 0
		corrs=[]
		array_stock = np.array(self.stock.as_matrix()[:,1])
		while k < len(array_stock)-1:

			path = np.array(self.paths[k])
			testpd = pd.DataFrame({"PATH":path,
								"STOCK": self.stock.as_matrix()[:,1]})
			corr = testpd["PATH"].astype("float64").corr(testpd["STOCK"].astype("float64"))
			corrs.append(corr)
			k+=1
		self.max_index = corrs.index(max(corrs))
		print(max(corrs))
		print(self.max_index) 
		#print(testpd.astype("float64").corr())
		#print(testpd["PATH"].astype("float64").corr(testpd["STOCK"].astype("float64")))
	def plottis(self):
		x = random.randrange(20)
		x = round(x)
		print(x, " XXXX")
		plt.plot(self.stock["DATE"].values, self.stock["STOCK"].values, "r-")
		plt.plot(self.stocks_model.time_grid, self.paths[:,:10], "--")
		plt.grid()
		plt.legend(["stock", "gbm"])
		plt.show()
	def strategy(self):
		pass














