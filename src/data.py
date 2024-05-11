import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Optional
from collections import defaultdict
from typing import Union

from numba import njit
from py_vollib.black_scholes.implied_volatility import implied_volatility

vec_find_vol_rat = np.vectorize(
    implied_volatility,
    doc="Vectorized function fro implied volatility computation, following the Let's be rational paper.",
)

# To represent volatility surfaces, we draw inspiration from Arthur Sepp's implementation of an option chain.
# The idea is to flatten the 3D data (like when pivoting a table) and have arrays maturity, strikes, prices.

@njit(cache=False, fastmath=True)
def erf(x: Union[float, np.ndarray]) -> float:
    """
    Error function approximation
    """
    z = np.abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * np.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+t*0.17087277)))))))))
    fcc = np.where(np.greater(x, 0.0), r, 2.0-r)
    return fcc

@njit(cache=False, fastmath=True)
def normal_cdf(x: float) -> float:
    return 1. - 0.5*erf(x/(np.sqrt(2.0)))

@njit(cache=False, fastmath=True)
def normal_pdf(x: float, mean: float = 0.0, var: float = 1.0) -> float:
    """
    Returns the normal density evaluated at point x.
    """
    vol = np.sqrt(var)
    return np.exp(-0.5 * np.square((x - mean) / vol)) / (vol * np.sqrt(2.0 * np.pi))


def ivs_to_prices(
    ivs: np.ndarray,
    ttms: np.ndarray,
    strikes: np.ndarray,
    underlying: Union[float, np.ndarray] = 100.0,
):
    """
    For a put.
    """
    d_1 = (np.log(underlying / strikes) + (ivs ** 2 / 2) * ttms) / (ivs * np.sqrt(ttms))
    d_2 = d_1 - ivs * np.sqrt(ttms)
    bs_prices = - underlying * norm.cdf(-d_1) + strikes * norm.cdf(-d_2)
    return bs_prices


def prices_to_ivs(
    prices: np.ndarray,
    ttms: np.ndarray,
    strikes: np.ndarray,
    forwards: np.ndarray,
    flags: np.ndarray,
):
    res = []
    for i in range(len(prices)):
        res.append(
            vec_find_vol_rat(
                price=prices[i],
                S=forwards[i],
                K=strikes[i],
                t=ttms[i],
                r=0,
                flag=flags[i],
            )
        )
    return res


class OptionChain:
    def __init__(
        self,
        ttms: np.ndarray,
        strikes: np.ndarray,
        forwards: np.ndarray,
        flags: np.ndarray,
        underlying: Union[float, np.ndarray] = 100.0,
        ivs: Optional[np.ndarray] = None,
        prices: Optional[np.ndarray] = None,
        bid_ivs: Optional[np.ndarray] = None,
        ask_ivs: Optional[np.ndarray] = None,
        bid_prices: Optional[np.ndarray] = None,
        ask_prices: Optional[np.ndarray] = None,
    ):
        self.underlying = np.array(underlying)
        self.ttms = np.array(ttms)
        self.strikes = np.array(strikes)
        self.forwards = np.array(forwards)
        self.flags = np.array(flags)

        if bid_ivs is not None and ask_ivs is not None:
            self.ivs = np.array((bid_ivs + ask_ivs) / 2)
        elif ivs is not None:
            self.ivs = np.array(ivs)
        elif bid_prices is not None and ask_prices is not None:
            self.prices = np.array((bid_prices + ask_prices) / 2)
        elif prices is not None:
            self.prices = np.array(prices)
        else:
            raise ValueError(
                "Incomplete option chain: please provide prices or implied volatility for all instruments."
            )

    def get_iv(self):
        if self.ivs is not None:
            return self.ivs
        else:
            self.ivs = np.zeros_like(self.prices)

            for i in range(len(self.prices)):
                self.ivs[i] = vec_find_vol_rat(
                    self.prices[i],
                    self.forwards[i],
                    self.strikes[i],
                    self.ttms[i],
                    0,
                    self.flags[i],
                )

        return self.ivs

    def ivs_to_prices(self):
        d_1 = (
            (
                np.log(self.underlying / self.strikes)
                + self.ivs**2 / 2 * np.sqrt(self.ttms)
            )
            / self.ivs
            * np.sqrt(self.ttms)
        )
        d_2 = d_1 - self.ivs * np.sqrt(self.ttms)
        bs_prices = self.underlying * normal_pdf(d_1) - self.strikes * normal_pdf(d_2)
        return bs_prices

    def prices_to_ivs(self):
        res = []
        for i in range(len(self.prices)):
            res.append(
                vec_find_vol_rat(
                    self.prices[i],
                    self.forwards[i],
                    self.strikes[i],
                    self.ttms[i],
                    0,
                    self.flags[i],
                )
            )
        return res

    def group_by_slice(self):
        slice_groups = defaultdict(lambda: {"strikes": [], "forwards": [], "flags": []})

        for ttm, strike, forward, flag in zip(
            self.ttms, self.strikes, self.forwards, self.flags
        ):
            slice_groups[ttm]["strikes"].append(strike)
            slice_groups[ttm]["forwards"].append(forward)
            slice_groups[ttm]["flags"].append(flag)

        for group in slice_groups.values():
            group["strikes"] = np.array(group["strikes"])
            group["forwards"] = np.array(group["forwards"])
            group["flags"] = np.array(group["flags"])

        return slice_groups

    def get_unique_ttms(self):
        """
        Returns an ordered list of the maturities for which we have data.
        """
        return sorted(set(self.ttms), key=lambda x: self.ttms.index(x))

    def get_vegas(self):
        """
        Return the chain vegas -- the Black Scholes greek at time t for a give maturity T is
        `S_t N'(d_1) \sqrt(T-t)`, where `d_1 = (\log(S_t/K) + (r + \sigma^2/2)(T-t)) / \sigma \sqrt(T-t)`
        and `N'(.)` is the standard normal pdf. Assumed `r=0`.
        """
        d_1 = (
            (
                np.log(self.underlying / self.strikes)
                + self.ivs**2 / 2 * np.sqrt(self.ttms)
            )
            / self.ivs
            * np.sqrt(self.ttms)
        )
        return self.underlying * normal_pdf(d_1) * np.sqrt(self.ttms)

    def plot_2d(self):
        """
        Scatter plot of the option surface assuming ivs are provided.
        """
        unique_ttms = np.unique(self.ttms)
        color_map = plt.cm.get_cmap("rainbow", len(unique_ttms))

        ttm_legend_added = {}

        fig, ax = plt.subplots()
        for ttm, strike, iv in zip(self.ttms, self.strikes, self.ivs):
            color_index = np.where(unique_ttms == ttm)[0][0]
            ax.scatter(
                strike,
                100 * iv,
                color=color_map(color_index),
                label=f"Maturity: {ttm}" if ttm not in ttm_legend_added else "",
            )
            ttm_legend_added[ttm] = True

        ax.set_xlabel("Strikes")
        ax.set_ylabel("IVs (in %)")
        ax.set_title("2D visualization of the option chain")
        ax.legend()

        fig.show()

    def plot_3d(self):
        unique_ttms = np.unique(self.ttms)
        color_map = plt.cm.get_cmap("rainbow", len(unique_ttms))

        ttm_legend_added = {}

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        for ttm, strike, iv in zip(self.ttms, self.strikes, self.ivs):
            color_index = np.where(unique_ttms == ttm)[0][0]
            ax.scatter(
                strike,
                ttm,
                100 * iv,
                color=color_map(color_index),
                label=f"Maturity: {ttm}" if ttm not in ttm_legend_added else "",
            )
            ttm_legend_added[ttm] = True

        ax.set_xlabel("Strikes")
        ax.set_zlabel("IVs (in %)")
        ax.set_ylabel("Time to Maturity")
        ax.set_title("3D visualization of the option chain")
        ax.legend()

        ax.view_init(elev=25, azim=-60)
        ax.dist = 11

        fig.show()
