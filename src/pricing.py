import numpy as np
import matplotlib.pyplot as plt
import scipy

from py_vollib.black_scholes.implied_volatility import implied_volatility
vec_find_vol_rat = np.vectorize(implied_volatility)

from src.models.base_model import BaseModel

def get_mc_iv():
    pass

def get_atm_skew(model: BaseModel, S0: float=100., plot: bool=True, fit: bool=True):
    hh = 0.001
    ttms = np.exp(np.linspace(np.log(1/52),np.log(2),20))
    ttms = np.append([1/252], ttms)
    strike_array = np.exp(np.array([-hh/2,hh/2]))*S0

    skew_calc = []
    for x in ttms:
        iv = model.get_iv(ttm=x, strikes=strike_array, forward=S0)
        skew_calc.append(iv)
    skew_calc = np.array(skew_calc)

    if fit:
        lin_regres_result = scipy.stats.linregress(np.log(ttms),np.log((skew_calc[:,0]-skew_calc[:,1])/hh))
        slope = lin_regres_result[0]
        intercept = lin_regres_result[1]
    
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        ax.scatter(ttms, (skew_calc[:,0]-skew_calc[:,1])/hh)
        if fit:
            ax.plot(ttms, np.exp(intercept)*ttms**(slope),'--r',label=r"$ \hat H \approx {:.2f}$".format(slope+0.5))
        ax.xlabel("$T$",fontsize=13)
        ax.set_ylabel("$\mathcal{S}_T$",fontsize=13)
        ax.legend()

        fig.show()


