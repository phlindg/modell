



from mylib import *
from model_stock import SModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import sys
import warnings
from sklearn import svm, linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


"""
TODO:
	Fixa long och short, ordentligt
	Fixa mean reversion.
	#Dubbelkolla att gen_trades faktiskt fungerar.
	En strategi som simulerar paths och longar/shortar
		beroende pa hur pathsen gar.


Kolla in: 
		http://www.eoddata.com/default.aspx
		SCIKIT LEARN!!!!!!

"""


class Strategy:
	#Klass med basicsaker, som att longa, korta
	#Sjalva strategierna ska generera signaler, 
	#denna ska hantera portfoljbyggande m.m.
	def __init__(self, assets, names,start, end):
		self.start = start
		self.end = end
		self.names = names
		self.dates = pd.date_range(start = self.start,
									end = self.end,
									freq = "B").to_pydatetime()
		mod_assets = []
		for asset in assets:
			asset.set_index("Date", inplace	= True)

			asset = asset[["Adj. Close"]]
			asset = asset[(asset.index >= start)
						& (asset.index <= end)]
			#asset.loc[:,"date"] = pd.to_datetime(asset.loc[:,"date"])
			cols = ["STOCK"]
			asset.index = pd.to_datetime(asset.index)
			asset.columns = cols
			asset["STOCK"] = asset["STOCK"].astype(float)
			 	
			mod_assets.append(asset)
		self.assets = mod_assets

	def long(self,asset,start = None, index = None):
		if start:
			s = asset.loc[start:,"STOCK"]
			
		if index:
			s = asset.iloc[index:,1]
		return s
	def short(self,asset,start = None, index = None):
		if start:
			s = np.array(asset.loc[start:,"STOCK"].values)
		if index:
			s = np.array(asset.iloc[index:, 1].values)
		i = 0
		r = []
		while i < len(s):
			if i == 0:
				r.append(s[0])
			val = s[i] - s[i-1]
			r_i = r[i-1] - val
			r.append(r_i)
			i+=1
		return r

	def order(self,asset,start,short = False):
		"""
		LATTARE ATT LAGGA TILL SHORT OCH LONG I EN FUNKTION!!
		"""
		s = asset.loc[start:,"STOCK"]
		i = 0
		r = []
		while i < len(s):
			if i == 0:
				r.append(s[0])
			val = s[i] - s[i-1]
			if short:
				r.append(r[i-1]-val)
			else:
				r.append(r[i-1]+val)

	def dP(self, asset, index):
		p1 = asset.iloc[index-1, 0]
		p2 = asset.iloc[index, 0]
		return p2 - p1

	def learn(self, model_end):

		clf = linear_model.LassoCV()
		start = str(self.assets[0].index[0])
		end = str(self.assets[0].index[-1])
		
		stock = SModel(self.assets[0], start, self.end, self.start, model_end, 100, "GBM")
		returns = stock.calc_returns().dropna()
		vol_ret = returns.std() * np.sqrt(len(self.assets[0].index))
		stock.create_market_env("me_stocks", vol_ret, dates="REAL")
		
		real = self.assets[0]["STOCK"].values
		dates = self.assets[0].index.values
		fake = stock.paths
		

		datum = pd.date_range(start = end, end = model_end, freq="B")
		k = len(datum)+8
		print("K: ", k)
		print(len(fake), len(real))
		fake_e = fake.astype("float64")
		real_e = real.astype("float64")
		fake_train, fake_test, real_train, real_test = train_test_split(fake_e,real_e,test_size = 0.4, random_state = 0)
		print(fake_train.shape, fake_test.shape, real_train.shape, real_test.shape)
		
		

		#ML STUFF

		clf.fit(list(fake_train), list(real_train))
		score = clf.score(fake_test, real_test)
		print(clf.alpha_)
		print(score)
		plt.plot(real_test)
		plt.plot(clf.predict(list(real_test)))
		
		plt.show()


	def plotem(self):
		plt.figure()
		for asset in self.assets:
			
			plt.plot(asset.iloc[:,0]/asset.iloc[0,0])
		plt.legend(self.names)
		plt.show()
	def plot_strat(self):
		plt.figure()
		dates = self.assets[0].index
		plt.plot(dates, self.strat_val)
		plt.plot(dates, self.real_val)
		plt.legend(["STRAT", "REAL"])
		plt.show()

	def set_signals(self, signals):
		i = 0
		if len(self.assets) == 1:
			self.assets[0]["SIGNALS"] = signals
		else:
			for asset in self.assets:
				
				asset["SIGNALS"] = signals[i]

				i+=1
	def gen_trades(self):
		rets = []
		
		for asset in self.assets:
			
			signs = asset["SIGNALS"]
			s = asset["STOCK"]
			ret = []
			short = False
			i = 0
			while i < len(signs):
				val = s.values[i] - s.values[i-1]
				if i == 0:
					ret.append(s.values[i])
				elif signs[i] == 1:
					ret.append(ret[i-1]+val)
					short = False
				elif signs[i] == -1:
					ret.append(ret[i-1]-val)
					short = True
				elif signs[i] == 0:
					if short == False:
						ret.append(ret[i-1]+val)
					if short == True:
						ret.append(ret[i-1]-val)
				
				i+=1
			rets.append(ret)		
		
		i = 0
		strat_val = list(map(sum, zip(*rets)))
		real = []
		for asset in self.assets:
			real.append(asset.loc[:,"STOCK"].values)
		real_val = list(map(sum, zip(*real)))
		self.strat_val = strat_val
		self.real_val = real_val


	
	def bollinger(self, vals):
		if len(self.assets) != 1:
			print("YOU NEED ONE STOCK FOR THIS ONE")
			sys.exit
		s = self.assets[0]["STOCK"]
		period_length = vals[0]
		factor = vals[1]
		print(factor)
		rmean = pd.rolling_mean(s, window = period_length, center = True)

		bol_up = rmean * (1+factor)
		bol_down = rmean * (1-factor)
		plt.plot(bol_up)
		plt.plot(bol_down)
		plt.plot(s)
		signals = []
		i = 0
		short = False
		while i < len(s):
			if s.values[i] > bol_up.values[i]:
				if short: 
					signals.append(0)
				else:
					signals.append(-1)
				short = True
			elif s.values[i] < bol_down.values[i]:
				if short:
					signals.append(1)
				else:
					signals.append(0)
				short = False
			else:
				signals.append(0)
			i += 1
		return signals

	def pairs_trade(self, vals):
		if len(self.assets) != 2:
			print("YOU NEED TWO STOCKS FOR THIS ONE")
			sys.exit
		s1 = self.assets[0]
		signals = []
		s2 = self.assets[1]
		dates = pd.date_range(start = self.start, 
							end = self.end,
							freq = "B")
		corrs = []
		i = 0
		period_length = vals[0]
		factor = vals[1]
		print(vals, period_length, factor)
		while i < len(s1):
			a = s1[period_length:i+period_length]["STOCK"]
			b = s2[period_length:i+period_length]["STOCK"]
			corr = np.corrcoef(a,b)[0][1]
			corrs.append(corr)
			i+=1

		corrs = np.array(corrs)[~np.isnan(corrs)]
		mean_corr = np.mean(corrs)
		corrs = np.append(corrs,mean_corr)
		corrs = np.append(corrs,mean_corr)
		a = []
		b = []
		i = 0
		
		if factor == "MEAN": 
			factor = mean_corr
		print(factor)
		
		while i < len(corrs):
			if corrs[i] < factor:
				if self.dP(s1, i) > 0:
					a.append(1)
					b.append(-1)
				elif self.dP(s1, i) < 0:
					a.append(-1)
					b.append(1)
				else:
					a.append(0)
					b.append(0)


			else:
				a.append(0)
				b.append(0)
			i+=1
		signals = [a,b]
		return signals
		#Kolla korrelation varje dag
		#Rakna ut mean
		#Nar det blir ifran mean, kolla vilken som stiger o vilken som sjunker
		#Korta den som sjunker, longa den som stiger
	def mean_rev(self, vals):
		if len(self.assets) != 1:
			print("YOU NEED ONE STOCK")
		s = self.assets[0]["STOCK"]
		longa = vals[0]
		short = vals[1]
		lmavg = pd.rolling_mean(s,longa)
		smavg = pd.rolling_mean(s,short)
		i = 0
		signals = []
		short = False
		while i < len(s):
			if i == 0:
				smavg.values[i] == s.values[i]
			if smavg.values[i-1] < lmavg.values[i] and smavg.values[i] >= lmavg.values[i]:
				short = True
				signals.append(-1)
			elif smavg.values[i-1] > lmavg.values[i] and smavg.values[i] <= lmavg.values[i]:
				short = False
				signals.append(1)
			else:
				signals.append(0)
			
			i+=1
		return signals


	def pathy(self, interval,num_paths):
		#For alla assets ska den:
			#Ta fram tidsintervall beroende pa "interval"
			#Genererea paths
			#Om 60% gar upp --> Longa
			#Om 60% gar ner --> korta
			#Aterupprepa tills man har gatt igenom hela tidenn.
		#Returna signalerna.
		pass

	def choose(self, name, *vals):
		dispatcher = {
		"bollinger": self.bollinger,
		"pairs_trade": self.pairs_trade,
		"mean_rev": self.mean_rev
		}
		return dispatcher[name](vals)