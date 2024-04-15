import numpy as np
from typing import Optional

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
        forward: np.ndarray,
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
        self.forward = forward
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

