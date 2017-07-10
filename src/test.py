from mylib import  *
from model_opt import OModel_JD
import pandas as pd


opts = pd.read_csv("data/opts.csv")
start = "2015-11-17"
end = "2016-06-25"
msft = OModel_JD("MSFT", opts, start, end)
msft.create_market_env("me_msft", 15.0, 1.0, 0.1, 0.5)
ranges = ((15.0, 25.0, 1.0),
			(-0.5, 0.5,  0.02),
			(0.0, 0.05, 0.01),
			(0.0, 0.05, 0.01))
msft.fit(ranges, (21.8530443006, -0.0443875543192, 0.00366927778959, 0.00873728220265))
msft.check()
