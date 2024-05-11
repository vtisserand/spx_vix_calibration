import numpy as np
import warnings
from scipy.special import gamma
from scipy.integrate import trapz, cumulative_trapezoid
from scipy.optimize import minimize
from typing import Optional
from collections import defaultdict
import mpmath as mp
from tqdm import tqdm
from mpmath import invertlaplace
from py_vollib.black_scholes.implied_volatility import implied_volatility

vec_find_vol_rat = np.vectorize(implied_volatility)

from src.models.base_model import BaseModel
from src.models.kernels import KernelFlavour
from src.data import OptionChain, prices_to_ivs, ivs_to_prices
from src.config import LOGGER, NB_DAYS_PER_YEAR


class qHeston(BaseModel):
    def __init__(
        self, initial_price: float = 100, kernel: KernelFlavour = KernelFlavour.ROUGH
    ):
        """
        The rough Bergomi model...
        """
        super().__init__(initial_price)
        self.kernel = kernel
        self.set_parameters()

    def __repr__(self):
        dynamics = "This is an instance of a quadratic Volterra Heston model."
        params = f"It has parameters a: {self.a}, b: {self.b}, c: {self.c}, H: {self.H}, eta: {self.eta}, eps: {self.eps}, rho: {self.rho}, fvc: {self.fvc}"
        return dynamics + "\n" + params

    def set_parameters(
        self,
        a: float = 0.384,
        b: float = 0.095,
        c: float = 0.0025,
        H: float = 0.08,
        eta: float = 0.7,
        eps: float = 1 / 52,
        rho: float = -0.6,
        fvc: float = 0.25,
    ):
        """
        Set of coherent dummy parameters to play around without market data, from the original rough Heston paper.
        """
        (
            self.a,
            self.b,
            self.c,
            self.H,
            self.eta,
            self.eps,
            self.rho,
            self.fvc,
        ) = a, b, c, H, eta, eps, rho, fvc
        # For the two-factor kernel, set aside for now:
        self.eta1, self.eta2 = eta, eta
        self.H1, self.H2 = H, H
        self.eps1, self.eps2 = eps, eps
        # Values to be stored for VIX computations after some trajectories are generated
        self.grid = None
        self.w1 = None

        self.kernel_std = {
            KernelFlavour.ROUGH: self._std_ji_rough,
            KernelFlavour.PATH_DEPENDENT: self._std_ji_path_dependent,
            KernelFlavour.ONE_FACTOR: self._std_ji_one_factor,
            KernelFlavour.TWO_FACTOR: self._std_ji_two_factor,
        }

    # We define the w_j^i for each kernel.
    def _std_ji_rough(self, tj: float, ti_s: np.ndarray | list) -> np.ndarray:
        std_ji = np.sqrt(
            (self.eta**2)
            * ((tj - ti_s[:-1]) ** (2 * self.H) - (tj - ti_s[1:]) ** (2 * self.H))
            / (2 * self.H)
        )
        return std_ji

    def _std_ji_path_dependent(self, tj: float, ti_s: np.ndarray | list) -> np.ndarray:
        if self.H == 0:
            std_ji = np.sqrt(
                (self.eta**2)
                * (np.log(tj - ti_s[:-1] + self.eps) - np.log(tj - ti_s[1:] + self.eps))
            )
        else:
            std_ji = np.sqrt(
                (self.eta**2)
                * (
                    (tj - ti_s[:-1] + self.eps) ** (2 * self.H)
                    - (tj - ti_s[1:] + self.eps) ** (2 * self.H)
                )
                / (2 * self.H)
            )
        return std_ji

    def _std_ji_one_factor(self, tj: float, ti_s: np.ndarray | list) -> np.ndarray:
        return (
            self.eta
            * self.eps**self.H
            * np.sqrt(
                (
                    np.exp((2 * self.H - 1) * (tj - ti_s[:-1]) / self.eps)
                    - np.exp((2 * self.H - 1) * (tj - ti_s[1:]) / self.eps)
                )
                / (2 * self.H - 1)
            )
        )

    def _std_ji_two_factor(self, tj: float, ti_s: np.ndarray | list) -> np.ndarray:
        acst = (self.eta1**2) * (self.eps1 ** (2 * self.H1)) / (2 * self.H1 - 1)
        bcst = (self.eta2**2) * (self.eps2 ** (2 * self.H2)) / (2 * self.H2 - 1)
        gamma = (1 / self.eps1) * (self.H1 - 0.5) + (1 / self.eps2) * (self.H2 - 0.5)
        if gamma == 0:
            ccst = (
                2
                * self.eta1
                * self.eta2
                * (self.eps1 ** (self.H1 - 0.5))
                * (self.eps2 ** (self.H2 - 0.5))
            )
        else:
            ccst = (
                2
                * self.eta1
                * self.eta2
                * (self.eps1 ** (self.H1 - 0.5))
                * (self.eps2 ** (self.H2 - 0.5))
                / gamma
            )

        a_ji = acst * (
            np.exp((2 * self.H1 - 1) * (tj - ti_s[:-1]) / self.eps1)
            - np.exp((2 * self.H1 - 1) * (tj - ti_s[1:]) / self.eps1)
        )
        b_ji = bcst * (
            np.exp((2 * self.H2 - 1) * (tj - ti_s[:-1]) / self.eps2)
            - np.exp((2 * self.H2 - 1) * (tj - ti_s[1:]) / self.eps2)
        )
        if gamma == 0:
            c_ji = ccst * (ti_s[1:] - ti_s[:-1])
        else:
            c_ji = ccst * (
                np.exp(gamma * (tj - ti_s[:-1])) - np.exp(gamma * (tj - ti_s[1:]))
            )

        return np.sqrt(a_ji + b_ji + c_ji)

    def _R_bar_rough(self, T: float, n_terms: int = 100):
        alpha = -self.a * self.eta**2 * gamma(2 * self.H)
        return (1 / (2 * self.H)) * sum(
            ((-1) ** k)
            * (alpha ** (k + 1))
            * (T ** (2 * self.H * (k + 1)))
            / ((k + 1) * gamma(2 * self.H * (k + 1)))
            for k in range(n_terms)
        )

    def _laplace_k_tilde_path_dependent(self, u):
        if u == 0:
            return (
                self.a * (self.eta**2) * (self.eps ** (2 * self.H)) / (2 * self.H)
                if self.H > 0
                else mp.mpf("+inf")
            )
        return (
            -self.a
            * (self.eta**2)
            * mp.exp(u * self.eps)
            * (1 / (u ** (2 * self.H)))
            * mp.gammainc(2 * self.H, a=u * self.eps)
        )

    def _calculate_resolvent_path_dependent(self, tau):
        resolvent = []
        for t in tau:
            resolvent.append(
                mp.invertlaplace(
                    lambda u: 1 / (1 + (1 / self._laplace_k_tilde_path_dependent(u))),
                    t,
                    method="cohen",
                )
            )
        return np.array(resolvent)

    def _R_bar_path_dependent(self, t):
        tt = np.linspace(0, 1 / 2, 1000)
        tt = 1e-20 + np.power(tt, 10)
        tt = np.concatenate(
            [tt, np.linspace(tt[-1], t[-1], max(1000, len(t)))[1:]], axis=0
        )
        resolvent = self._calculate_resolvent_path_dependent(tt)
        integrale_resolvent = cumulative_trapezoid(resolvent, tt)
        integrale_resolvent = np.concatenate([[0], integrale_resolvent], axis=0)
        integrale_resolvent = np.array([float(el) for el in integrale_resolvent])
        integrale_resolvent = np.interp(t, tt, integrale_resolvent)
        return integrale_resolvent

    def _R_bar_one_factor(self, t):
        cst = -self.a * (self.eta**2) * (self.eps ** (2 * self.H - 1))
        mul = -(2 * self.H - 1) / self.eps
        return (cst / (cst + mul)) * (1 - np.exp(-(cst + mul) * t))

    def _calculate_laplace_transform_exp(self, u, cst, mul):
        if cst == 0:
            return 0
        if mp.re(u) <= mul:
            return cst * mp.mpf("+inf")
        return cst / (u - mul)

    def _laplace_k_tilde_two_factor(self, u, cst1, cst2, cst3, mul1, mul2, mul3):
        laplace_k_tilde1 = self._calculate_laplace_transform_exp(u, cst1, mul1)
        laplace_k_tilde2 = self._calculate_laplace_transform_exp(u, cst2, mul2)
        laplace_k_tilde3 = self._calculate_laplace_transform_exp(u, cst3, mul3)
        return laplace_k_tilde1 + laplace_k_tilde2 + laplace_k_tilde3

    def _calculate_resolvent_two_factor(self, tau):
        cst1 = -self.a * (self.eta1**2) * (self.eps1 ** (2 * self.H1 - 1))
        cst2 = (
            -2
            * self.a
            * self.eta1
            * self.eta2
            * (self.eps1 ** (self.H1 - 0.5))
            * (self.eps2 ** (self.H2 - 0.5))
        )
        cst3 = -self.a * (self.eta2**2) * (self.eps2 ** (2 * self.H2 - 1))
        mul1 = (2 * self.H1 - 1) / self.eps1
        mul2 = (1 / self.eps1) * (self.H1 - 0.5) + (1 / self.eps2) * (self.H2 - 0.5)
        mul3 = (2 * self.H2 - 1) / self.eps2
        resolvent = []
        for t in range(tau):
            # With 'talbot', the resolvent diverges to infinity as t gets closer to 0.
            # 'cohen' is more appropriate for dealing with such singularities.
            resolvent.append(
                mp.invertlaplace(
                    lambda u: 1
                    / (
                        1
                        + (
                            1
                            / self.laplace_k_tilde_two_factor(
                                u, cst1, cst2, cst3, mul1, mul2, mul3
                            )
                        )
                    ),
                    t,
                    method="cohen",
                )
            )
        return np.array(resolvent)

    def _R_bar_two_factor(self, t):
        t_wth_zero = np.concatenate([[1e-20], t[1:]], axis=0)
        resolvent = self._calculate_resolvent_two_factor(t_wth_zero)
        integrale_resolvent = cumulative_trapezoid(resolvent, t_wth_zero)
        integrale_resolvent = np.concatenate([[0], integrale_resolvent], axis=0)
        integrale_resolvent = np.array([float(el) for el in integrale_resolvent])
        return integrale_resolvent

    def generate_paths(
        self,
        n_steps: int,
        length: int = 1,
        n_sims: int = 1,
    ):
        # Uncorrelated brownians
        w1, w2 = (
            np.random.normal(0, 1, (int(n_steps * length), n_sims)),
            np.random.normal(0, 1, (int(n_steps * length), n_sims)),
        )
        tt = np.linspace(0, length, int(n_steps * length) + 1)

        # Initiate rough Heston Z process and quadratic form instantaneous variance V
        dt = tt[1] - tt[0]  # Uniform grid
        N_sims = w1.shape[1]
        Z = np.zeros(N_sims).reshape(1, -1)
        V = np.zeros(N_sims).reshape(1, -1)
        Z[0] = self.fvc
        V[0] = self.a * (self.fvc - self.b) ** 2 + self.c

        for j in tqdm(range(int(n_steps * length))):
            tj = tt[j + 1]
            ti_s = tt[: j + 2]

            if self.kernel in self.kernel_std:
                std_ji = self.kernel_std[self.kernel](tj, ti_s)
            else:
                warnings.warn(
                    "Unrecognized kernel type. Please provide a valid kernel type from KernelFlavour enum.",
                    UserWarning,
                )
                raise ValueError("Invalid kernel type provided.")

            Z_temp = self.fvc + np.sum(
                std_ji.reshape(-1, 1) * w1[: j + 1] * np.sqrt(V[: j + 1]), axis=0
            )
            V_temp = self.a * ((Z_temp - self.b) ** 2) + self.c
            Z = np.append(Z, Z_temp.reshape(1, -1), axis=0)
            V = np.append(V, V_temp.reshape(1, -1), axis=0)

        # Correlate the brownians with rho
        correlated_brownian = self.rho * w1 + np.sqrt(1 - self.rho**2) * w2
        logSt_increment = -0.5 * dt * V[:-1, :] + np.sqrt(dt) * np.multiply(
            np.sqrt(V[:-1, :]), correlated_brownian
        )
        logSt_increment = np.concatenate(
            [np.full((1, n_sims), np.log(self.initial_price)), logSt_increment], axis=0
        )
        logSt = np.cumsum(logSt_increment, axis=0)
        St = np.exp(logSt)

        self.grid = tt  # Needed to match grids with VIX levels computations
        self.w1 = w1

        return St, V

    def compute_vix(
        self,
        t,
        tau,
        delta,
        V,
        n_steps: int,
        n_sims: int,
    ):
        """
        Parameters:
          t: times at which we want to compute VIX levels, e.g. for 6 months options MC, t=[1/2],
          tau: discretization grid, matches the one of the Euler scheme to generathe the sample paths,
          delta: how far is the VIX looking. Classicly 1/12 vut for VIX3M 1/4 for instance,
          V: the variance process,
        """
        tt = self.grid

        if self.kernel == KernelFlavour.ROUGH:
            int_res = self._R_bar_rough(T=tau)
        elif self.kernel == KernelFlavour.PATH_DEPENDENT:
            int_res = self._R_bar_path_dependent(t=tau)
        elif self.kernel == KernelFlavour.ONE_FACTOR:
            int_res = self._R_bar_one_factor(t=tau)
        elif self.kernel == KernelFlavour.TWO_FACTOR:
            int_res = self._R_bar_two_factor(t=tau)
        else:
            warnings.warn(
                "Unrecognized kernel type. Please provide a valid kernel type from KernelFlavour enum.",
                UserWarning,
            )
            raise ValueError("Invalid kernel type provided.")

        int_res = 1 - np.flip(int_res)
        int_res = int_res.reshape((-1, 1))

        vix = []
        for time_t in tqdm(
            t
        ):  # Nested loop as we integrate of the forward variance at each time step to get VIX levels
            j = int(n_steps * time_t)
            f = []
            for time_tau in tau:
                k = int(n_steps * time_tau)
                tj = tt[j + k + 1]
                ti_s = tt[: j + 2]

                if self.kernel in self.kernel_std:
                    std_ji = self.kernel_std[self.kernel](tj, ti_s)
                else:
                    warnings.warn(
                        "Unrecognized kernel type. Please provide a valid kernel type from KernelFlavour enum.",
                        UserWarning,
                    )
                    raise ValueError("Invalid kernel type provided.")

                g = self.fvc + np.sum(
                    std_ji.reshape(-1, 1) * self.w1[: j + 1] * np.sqrt(V[: j + 1]),
                    axis=0,
                )
                f.append((self.a * (g - self.b) ** 2 + self.c).reshape((1, -1)))
            f = np.concatenate(f, axis=0)
            integrand = int_res * f

            l = []
            for i in range(n_sims):
                l.append(trapz(integrand[:, i], tau))
            vix_temp = np.array(l).reshape((1, -1))
            vix.append(vix_temp)

        vix = np.concatenate(vix, axis=0)
        vix = np.sqrt((10000 / delta) * vix)
        return vix

    def get_prices(
        self,
        prices: np.ndarray,
        ttm: float,
        n_steps: int,
        strikes: np.ndarray,
        forward: float,
    ):
        prices_ttm = prices[int(ttm * n_steps)]
        opt_prices = np.mean(
            np.maximum(
                strikes
                - np.repeat(prices_ttm.reshape((-1, 1)), repeats=len(strikes), axis=1),
                0,
            ),
            axis=0,
        )
        return opt_prices

    def get_iv(
        self,
        prices: np.ndarray,
        ttm: float,
        n_steps: int,
        strikes: np.ndarray,
        forward: float,
    ):
        prices_ttm = prices[int(ttm * n_steps)]

        intrinsic_value = np.maximum(
            np.array(strikes) - np.array([forward] * len(strikes)), 0.0
        )
        opt_prices = np.mean(
            np.maximum(
                strikes
                - np.repeat(prices_ttm.reshape((-1, 1)), repeats=len(strikes), axis=1),
                0,
            ),
            axis=0,
        )
        opt_prices = np.where(
            opt_prices >= intrinsic_value, opt_prices, intrinsic_value
        )  # Penalize arbitrageable values.

        iv_mid = vec_find_vol_rat(opt_prices, forward, strikes, ttm, 0, "p")
        return iv_mid

    def get_iv_vix(
        self,
        prices_ttm: np.ndarray,
        ttm: np.ndarray,
        strikes: np.ndarray,
        future: float,
    ):
        intrinsic_value = np.maximum(
            np.array(strikes) - np.array([future] * len(strikes)), 0.0
        )
        opt_prices = np.mean(
            np.maximum(
                strikes
                - np.repeat(prices_ttm.reshape((-1, 1)), repeats=len(strikes), axis=1),
                0,
            ),
            axis=0,
        )
        opt_prices = np.where(
            opt_prices >= intrinsic_value, opt_prices, intrinsic_value
        )  # Penalize arbitrageable values.

        iv_mid = vec_find_vol_rat(opt_prices, future, strikes, ttm, 0, "p")
        return iv_mid

    def fit(
        self,
        option_chain: OptionChain,
        is_vega_weighted: bool = False,
        verbose: Optional[bool] = False,
    ):
        """
        Method to fit a qHeston model on a single surface. Default minimizes MSE on implied volatilities while if
        is_vega_weighted is set to true minimizes MSE on prices, applying inverse vega weights such that the wings
        still matter and the smile does not fade away.
        """
        n_steps = NB_DAYS_PER_YEAR  # Change to ensure less biased MC estimator
        n_sims = 100000

        if is_vega_weighted:
            market_data = ivs_to_prices(
                ivs=option_chain.ivs,
                ttms=option_chain.ttms,
                strikes=option_chain.strikes,
                underlying=option_chain.underlying,
            )
        else:
            market_data = option_chain.get_iv()

        errs = []

        def objective(params: np.ndarray, *args) -> float:
            if verbose:
                print(f"params: {params}")
            self.set_parameters(*params)
            # Sample paths with a buffer for long maturities
            prices, _ = self.generate_paths(
                n_steps=n_steps, length=1.1 * max(option_chain.ttms), n_sims=n_sims
            )

            # Here we group the computations by slices (options accros different strikes for the same maturity).
            slices = option_chain.group_by_slice()
            model_data = []
            for ttm, slice_data in slices.items():
                if is_vega_weighted:
                    model_data.append(
                        self.get_prices(
                            prices,
                            ttm,
                            n_steps,
                            slice_data["strikes"],
                            slice_data["forwards"][0],
                        )
                    )
                else:
                    model_data.append(
                        self.get_iv(
                            prices,
                            ttm,
                            n_steps,
                            slice_data["strikes"],
                            slice_data["forwards"][0],
                        )
                    )
            if verbose:
                print(f"market: {100*np.array(market_data)}")
                print(f"model: {100*np.array(model_data).flatten()}")

            if is_vega_weighted:
                error = np.sum(
                    np.square(np.array(model_data).flatten() - np.array(market_data))
                    / (option_chain.get_vegas() / 100)
                )
            else:
                error = np.sum(
                    np.square(
                        100 * np.array(model_data).flatten()
                        - 100 * np.array(market_data)
                    )
                )
            errs.append(error)

            if verbose:
                print(f"error: {error}")

            return error

        init_guess = np.array([0.1, 0.1, 0.05, 0.25, 0.7, 1 / 52, -0.9, 0.5])
        LOGGER.info(
            f"Initiating single surface calibration with parameters: {init_guess}."
        )

        res = minimize(
            objective,
            init_guess,
            args=(option_chain,),
            method="nelder-mead",
        )

        return res.x, errs

    def fit_joint(
        self,
        spx_option_chain: OptionChain,
        vix_option_chain: OptionChain,
        vix_futures: np.ndarray,
        is_vega_weighted: bool = False,
        verbose: Optional[bool] = False,
    ):
        """
        Method to fit a qHeston model on SPX and VIX surfaces (joint calibration problem).
        Default minimizes MSE on implied volatilities while if is_vega_weighted is set to true
        minimizes MSE on prices, applying inverse vega weights such that the wings still matter
        and the smile does not fade away.
        """
        n_steps = NB_DAYS_PER_YEAR  # Change to ensure less biased MC estimator
        n_sims = 100000

        if is_vega_weighted:
            spx_market_data = ivs_to_prices(
                ivs=spx_option_chain.ivs,
                ttms=spx_option_chain.ttms,
                strikes=spx_option_chain.strikes,
                underlying=spx_option_chain.underlying,
            )
            vix_market_data = ivs_to_prices(
                ivs=vix_option_chain.ivs,
                ttms=vix_option_chain.ttms,
                strikes=vix_option_chain.strikes,
                underlying=vix_option_chain.underlying,
            )
        else:
            spx_market_data = spx_option_chain.get_iv()
            vix_market_data = vix_option_chain.get_iv()

        def objective_joint(params: np.ndarray, args: np.ndarray) -> float:
            print(f"params: {params}")
            self.set_parameters(*params)
            # Sample paths with a buffer for long maturities
            S, V = self.generate_paths(
                n_steps=n_steps, length=1.2 * max(max(spx_option_chain.ttms), max(vix_option_chain.ttms)), n_sims=n_sims
            )

            # SPX error
            slices = spx_option_chain.group_by_slice()
            spx_model_data = []
            for ttm, slice_data in slices.items():
                if is_vega_weighted:
                    spx_model_data.append(
                        self.get_prices(
                            self,
                            ttm,
                            n_steps,
                            slice_data["strikes"],
                            slice_data["forwards"][0],
                        )
                    )
                else:
                    spx_model_data.append(
                        self.get_iv(
                            S,
                            ttm,
                            n_steps,
                            slice_data["strikes"],
                            slice_data["forwards"][0],
                        )
                    )

            if is_vega_weighted:
                spx_error = np.sum(
                    np.square(np.array(spx_model_data).flatten() - np.array(spx_market_data))
                    / (spx_option_chain.get_vegas() / 100)
                )
            else:
                spx_error = np.sum(
                    np.square(
                        100 * np.array(spx_model_data).flatten()
                        - 100 * np.array(spx_market_data)
                    )
                )

            # VIX data
            vix_ttms = vix_option_chain.get_unique_ttms()
            tt = self.grid
            delta = 1 / 12  # VIX sees 1 month further
            taus = np.concatenate([[1e-5], tt[1 : int(n_steps * delta)]])
            vix = self.compute_vix(
                t=vix_ttms, tau=taus, delta=delta, V=V, n_steps=n_steps, n_sims=n_sims
            )

            # VIX futures error
            vix_fut_model = vix.mean(axis=1)
            vix_fut_error = 10 * np.sum(
                np.square(np.array(vix_fut_model) - np.array(vix_futures))
            )

            # VIX error
            vix_slices = vix_option_chain.group_by_slice()
            vix_model_data = []
            for i, data in enumerate(vix_slices.items()):
                ttm, slice_data = data[0], data[1]

                if is_vega_weighted:
                    vix_model_data.append(
                        self.get_prices(
                            self,
                            ttm,
                            n_steps,
                            slice_data["strikes"],
                            slice_data["forwards"][0],
                        )
                    )
                else:
                    vix_model_data.append(
                    self.get_iv_vix(
                        prices_ttm=vix,
                        ttm=ttm,
                        strikes=slice_data["strikes"],
                        future=vix_fut_model[i],
                    )
                )
                    
            if is_vega_weighted:
                vix_error = np.sum(
                    np.square(np.array(vix_model_data).flatten() - np.array(vix_market_data))
                    / (vix_option_chain.get_vegas() / 100)
                )
            else:
                vix_error = np.sum(
                    np.square(
                        100 * np.array(vix_model_data).flatten()
                        - 100 * np.array(vix_market_data)
                    )
                )

            if verbose:
                print(
                    f"Errors: SPX {spx_error}, VIX futures {vix_fut_error}, VIX {vix_error}."
                )

            return spx_error + vix_fut_error + vix_error

        init_guess = np.array([0.1, 0.1, 0.05, 0.25, 0.7, 1 / 52, -0.9, 0.5])
        LOGGER.info(
            f"Initiating joint calibration with parameters: {init_guess}."
        )

        res = minimize(
            objective_joint,
            init_guess,
            args=None,
            method="nelder-mead",
        )

        return res.x
