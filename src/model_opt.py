from mylib import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import calendar

#1. Get and modify data
msft = pd.read_csv("data/msft.csv")
aapl = pd.read_csv("data/aapl.csv")

def third_friday(date):
    day = 21 - (calendar.weekday(date.year, date.month, 1)+2)%7
    return dt.datetime(date.year, date.month, day)
class OModel_JD:
	def __init__(self, stock_name, opts,start, end):
		
		self.start = start
		self.end = end
		opts = opts[opts.ticker == stock_name]
		data = opts[["stkPx", "expirDate", "strike", "cValue", "trade_date"]]
		cols = ["STOCK PRICE", "MATURITY", "STRIKE", "CALL VALUE", "DATE"]
		data.columns = cols
		test = []
		for opt in data["MATURITY"]:
			test.append(int(opt[5:7]))
		data.insert(5, "EXP_MONTH", test)
		data = data[(data["MATURITY"] < end)]
		self.data = data

		

	def create_market_env(self,name, lamb,mu,delt,vol):
		third_fridays = {}
		data = self.data
		for month in set(data["EXP_MONTH"]):
			third_fridays[month] = third_friday(dt.datetime(2016,month,1))
		self.pricing_date = dt.datetime.strptime(self.start, "%Y-%m-%d")
		final_date = dt.datetime.strptime(self.start, "%Y-%m-%d")
		maturity = third_fridays[6]
		maturity_str = str(maturity)
		tol = .2
		initial_value = data.iloc[0]["STOCK PRICE"]
		
		option_selection = data[(data["DATE"].values == self.start)
                        & (data["MATURITY"].values == maturity_str[0:10])
                        & (data["STRIKE"].values > (1-tol)*initial_value)
                        & (data["STRIKE"].values < (1+tol)*initial_value)]

		print(option_selection)


		me_stock = market_env(name, self.pricing_date)
		me_stock.add_constant("initial_value", initial_value)
		me_stock.add_constant("final_date", maturity)
		me_stock.add_constant("currency", "USD")
		me_stock.add_constant("frequency", "B")
		me_stock.add_constant("paths", 10000)
		csr = constant_short_rate("csr", 0.01)
		me_stock.add_curve("discount_curve", csr)
		me_stock.add_constant("lambda", lamb)
		me_stock.add_constant("mu", mu)
		me_stock.add_constant("delta", delt)
		me_stock.add_constant("volatility", vol)
		me_stock.add_constant("strike", initial_value)
		me_stock.add_constant("maturity", maturity)
		stocks_model = jump_diffusion("stocks_model", me_stock)
		
		payoff_func = "np.maximum(maturity_value - strike, 0)"
		
		stocks_eu_call = valuation_eu("stocks_eu_call", stocks_model, me_stock, payoff_func)
		option_models = {}
		
		for option in option_selection.index:
			strike = option_selection["STRIKE"].ix[option]
			print(strike)
			me_stock.add_constant("strike", strike)
			option_models[option] = valuation_eu(
				"eur_call_%d" % strike,
				stocks_model,
				me_stock,
				payoff_func)
		self.option_models = option_models
		self.me_stock = me_stock
		self.stocks_model = stocks_model
		self.option_selection = option_selection
		self.option_models = option_models
	def calc_model_values(self,p0):
		lamb, mu, delt, vol = p0

		self.stocks_model.update(lamb = lamb, mu = mu, delt = delt, volatility = vol)
		model_values = {}
		#print(self.option_models)
		for option in self.option_models:
			model_values[option] = self.option_models[option].present_value(fixed_seed = True)
		
		return model_values
	def mean_squared_error(self,p0):
		i = 0
		model_values = np.array(list(self.calc_model_values(p0).values()))
		
		
		market_values = self.option_selection["CALL VALUE"].values
		option_diffs = model_values - market_values

		MSE = np.sum(option_diffs ** 2)/len(option_diffs)

		if i % 20 == 0:
			if i == 0:
				
				print("%4s %6s %6s %6s %6s --> %6s" % ("i","lambda","mu", "delta", "vola", "MSE"))
			print("%4s %6s %6s %6s %6s --> %6s" % (i,p0[0],p0[1],p0[2],p0[3], MSE))
		i+=1
		return MSE
	def fit(self,ranges, already_known = None):
		if already_known == None:
			#opt_global = spo.brute(self.mean_squared_error,
									#ranges,
									#finish = None)
			opt_global = (21.12, -0.06124, 0.0034168, 0.008175)
			opt_local = spo.fmin(self.mean_squared_error, 
		  							opt_global,
		  							xtol = 0.0001, ftol = 0.0001,
		  							maxiter = 1000, maxfun = 1000)
			self.p0 = opt_local
		else:
			self.p0 =  already_known
	def check(self):
		self.option_selection.loc[:,"MODEL"] = np.array(list(self.calc_model_values(self.p0).values()))
		self.option_selection.loc[:,"ERRORS"] = self.option_selection["MODEL"] - self.option_selection["CALL VALUE"]
		print(self.option_selection[["MODEL", "CALL VALUE", "ERRORS"]])
	def strategy(self):
		pass

	def create_portfolio(self):
		pass












