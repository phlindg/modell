import scipy.optimize as spo

def myopt(mse, ranges):
    opt_global = spo.brute(mse, ranges,finish = None)
    print("OPT GLOBAL ", opt_global)
    opt_local = spo.fmin(mse, opt_global, xtol = 0.001, ftol = 0.0001,
        maxiter = 1000, maxfun = 1000)
    return opt_local
