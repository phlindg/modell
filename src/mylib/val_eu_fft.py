

from mylib.valuation_class import valuation_class
import math
import numpy as np
from numpy.fft import *


def M76_characteristic_function(u, x0, T, r, sigma, lamb, mu, delta):
	omega = x0/T + r - 0.5*sigma**2 -lamb * (np.exp(mu+0.5*delta**2)-1)
	value = np.exp((1j * u* omega - 0.5*u**2 * sigma**2 + lamb * (np.exp(1j * u * mu - u**2 * delta**2 * 0.5)-1))*T)
	return value
def M76_value_call_FFT(s0, K, T, r, sigma, lamb, mu, delta):
	"""
	sigma: float
		vol factor in diffusion term
	lamb: float
		jump intensity
	mu: float
		expected jump size <-- wow
	delta:  float
		std of jump
	"""

	k = math.log(K/s0)
	x0 = math.log(s0/s0)
	g = 2 #factor to increase accuracy
	N = g*4096
	exp = (g*150.)**(-1)
	eta = 2*math.pi/(N*eps)
	b = 0.5*N*eps - k
	u = np.arange(1, N+1, 1)
	vo = eta*(u-1)
	#Modifications to ensure integrability
	if s0 >= 0.95*K: #ITM case
		alpha = 1.5
		v = vo - (alpha+1)*1j
		mod_char_fun_1 = math.exp(-r * T) * (1 / (1 + 1j * (vo - 1j * alpha)) - math.exp(r * T) / (1j * (vo - 1j * alpha)) - M76_characteristic_function(v, x0, T, r, sigma, lamb, mu, delta) / ((vo -1j*alpha)**2-1j*(vo-1j*alpha)))
		



class valuation_fft(valuation_class):

	def generate_payoff(self, fixed_seed = False):
		try:
			strike = self.strike
		except:
			pass
		s0 = self.initial_value
		K = self.strike
		T = self.maturity
		r = self.discount_curve
		sigma = self.volatility
		lamb = self.lamb
		mu = self.mu
		delta = self.delt

