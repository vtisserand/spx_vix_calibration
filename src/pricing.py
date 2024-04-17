import numpy as np
import matplotlib.pyplot as plt
import scipy

from src.models.base_model import BaseModel
from src.config import NB_DAYS_PER_YEAR

def get_atm_skew(model: BaseModel, prices: np.ndarray, S0: float=100., plot: bool=True, fit: bool=True):
    hh = 1e-4
    ttms = np.exp(np.linspace(np.log(1/52),np.log(2),20))
    ttms = np.append([1/252], ttms)
    strike_array = np.exp(np.array([-hh/2,hh/2]))*S0

    skew_calc = []
    for x in ttms:
        iv = model.get_iv(prices=prices, ttm=x, strikes=strike_array, forward=S0)
        skew_calc.append(iv)
    skew_calc = np.array(skew_calc) # A len(ttms) x 2 array to make finite differences derivative approximation

    print(skew_calc)

    if fit:
        lin_regres_result = scipy.stats.linregress(np.log(ttms),np.log((skew_calc[:,0]-skew_calc[:,1])/hh))
        slope = lin_regres_result[0]
        intercept = lin_regres_result[1]
    
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(ttms, (skew_calc[:,0]-skew_calc[:,1])/hh, c='purple')
        if fit:
            ax.plot(ttms, np.exp(intercept)*ttms**(slope),'--r',label=r"$ \hat H \approx {:.2f}$".format(slope+0.5))
        ax.set_xlabel("$T$",fontsize=13)
        ax.set_ylabel("$\mathcal{S}_T$",fontsize=13)
        ax.legend()

        fig.show()

def plot_spx_surface(model: BaseModel, S0: float=100.):
    ttm_array = np.array([1/52,2/52,1/12,2/12,3/12,6/12,1,1.5,2])
    ttm_array_name = np.array(['1w','2w','1m','2m','3m','6m','1y','18m','2y'])

    # Log-moneyness ranges
    def spx_range_rule(ttm):
        if ttm<=2/52:
            lm_range = [-0.2,0.05]
        elif ttm <=1/12:
            lm_range = [-0.3,0.07]
        elif ttm <= 2/12:
            lm_range = [-0.4,0.1]
        elif ttm <= 3/12:
            lm_range = [-0.5,0.15]
        elif ttm <= 6/12:
            lm_range = [-0.6,0.15]
        elif ttm <= 12/12:
            lm_range = [-0.7,0.2]
        else:
            lm_range = [-0.8,0.3]
        return lm_range
    
    prices, _ = model.generate_paths(n_steps=NB_DAYS_PER_YEAR, length=1.1*max(ttm_array), n_sims=300000)

    ivs = []
    logks = []
    for ttm in ttm_array:
        logk_lu = spx_range_rule(ttm)
        logk = np.linspace(logk_lu[0], logk_lu[1], 50)
        strike_array = np.exp(logk)*S0
        smile = model.get_iv(prices=prices, ttm=ttm, strikes=strike_array, forward=S0)
        ivs.append(smile)
        logks.append(logk)

    fig, ax = plt.subplots()
    for i in range(len(ttm_array)):
        ax.plot(logks[i], ivs[i], label = ttm_array_name[i])
    ax.set_xlabel("log moneyness $\log(K/S_0)$",fontsize=13)
    ax.set_ylabel("$\sigma_{BS}(T,k)$",fontsize=13)
    ax.legend()
    fig.show()