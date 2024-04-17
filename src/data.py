import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from collections import defaultdict

from py_vollib.black_scholes.implied_volatility import implied_volatility

vec_find_vol_rat = np.vectorize(
    implied_volatility,
    doc="Vectorized function fro implied volatility computation, following the Let's be rational paper.",
)

# To represent volatility surfaces, we draw inspiration from Arthur Sepp's implementation of an option chain.
# The idea is to flatten the 3D data (like when pivoting a table) and have arrays maturity, strikes, prices.


class OptionChain:
    def __init__(
        self,
        ttms: np.ndarray,
        strikes: np.ndarray,
        forwards: np.ndarray,
        flags: np.ndarray,
        ivs: Optional[np.ndarray] = None,
        prices: Optional[np.ndarray] = None,
        bid_ivs: Optional[np.ndarray] = None,
        ask_ivs: Optional[np.ndarray] = None,
        bid_prices: Optional[np.ndarray] = None,
        ask_prices: Optional[np.ndarray] = None,
    ):
        self.ttms = ttms
        self.strikes = strikes
        self.forwards = forwards
        self.flags = flags

        if bid_ivs is not None and ask_ivs is not None:
            self.ivs = (bid_ivs + ask_ivs) / 2
        elif ivs is not None:
            self.ivs = ivs
        elif bid_prices is not None and ask_prices is not None:
            self.prices = (bid_prices + ask_prices) / 2
        elif prices is not None:
            self.prices = prices
        else:
            raise ValueError("Incomplete option chain: please provide prices or implied volatility for all instruments.")

    def get_iv(self):
        if self.ivs is not None:
            return self.ivs
        else:
            self.ivs = np.zeros_like(self.prices)
            
            for i in range(len(self.prices)):
                self.ivs[i] = vec_find_vol_rat(self.prices[i], self.forward[i], self.strikes[i], self.ttms[i], 0, self.flags[i])

        return self.ivs

    def group_by_slice(self):
        slice_groups = defaultdict(lambda: {"strikes": [], "forwards": [], "flags": []})
        
        for ttm, strike, forward, flag in zip(self.ttms, self.strikes, self.forwards, self.flags):
            slice_groups[ttm]["strikes"].append(strike)
            slice_groups[ttm]["forwards"].append(forward)
            slice_groups[ttm]["flags"].append(flag)
        
        for group in slice_groups.values():
            group["strikes"] = np.array(group["strikes"])
            group["forwards"] = np.array(group["forwards"])
            group["flags"] = np.array(group["flags"])
            
        return slice_groups
    
    def plot_2d(self):
        """
        Scatter plot of the option surface assuming ivs are provided.
        """
        unique_ttms = np.unique(self.ttms)
        color_map = plt.cm.get_cmap('rainbow', len(unique_ttms))

        ttm_legend_added = {}

        fig,ax=plt.subplots()
        for ttm, strike, iv in zip(self.ttms, self.strikes, self.ivs):
            color_index = np.where(unique_ttms == ttm)[0][0]
            ax.scatter(strike, 100*iv, color=color_map(color_index), label=f'Maturity: {ttm}' if ttm not in ttm_legend_added else '')
            ttm_legend_added[ttm] = True

        ax.set_xlabel('Strikes')
        ax.set_ylabel('IVs (in %)')
        ax.set_title('2D visualization of the option chain')
        ax.legend()

        fig.show()

    def plot_3d(self):
        unique_ttms = np.unique(self.ttms)
        color_map = plt.cm.get_cmap('rainbow', len(unique_ttms))

        ttm_legend_added = {}

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')

        for ttm, strike, iv in zip(self.ttms, self.strikes, self.ivs):
            color_index = np.where(unique_ttms == ttm)[0][0]
            ax.scatter(strike, ttm, 100*iv, color=color_map(color_index), label=f'Maturity: {ttm}' if ttm not in ttm_legend_added else '')
            ttm_legend_added[ttm] = True

        ax.set_xlabel('Strikes')
        ax.set_zlabel('IVs (in %)')
        ax.set_ylabel('Time to Maturity')
        ax.set_title('3D visualization of the option chain')
        ax.legend()

        ax.view_init(elev=25, azim=-60)
        ax.dist = 11

        fig.show()

