import numpy as np
import warnings
from scipy.special import gamma
from scipy.integrate import trapz, cumulative_trapezoid
import mpmath as mp
from mpmath import invertlaplace
from py_vollib.black_scholes.implied_volatility import implied_volatility

from src.models.base_model import BaseModel
from src.models.kernels import KernelFlavour, mittag_leffler


class qHeston(BaseModel):
    def __init__(self, initial_price: float = 100):
        """
        The rough Bergomi model...
        """
        super().__init__(initial_price)
        self.set_parameters()

    def __repr__(self):
        dynamics = "This is an instance of a quadratic Volterra Heston model."
        params = f"It has parameters a: {self.a}, b: {self.b}, c: {self.c}, H: {self.H}, eta: {self.eta}, eps: {self.eps}, rho: {self.rho}, fvc: {self.fvc}"
        return dynamics + '\n' + params

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
            KernelFlavour.TWO_FACTOR: self._std_ji_two_factor
        }

    def fit(
        self,
        strikes: np.ndarray | list[float],
        prices: np.ndarray | list[float],
        forward_price: float = 100,
        maturity: float = 1,
    ):
        """ """
        pass

    # We define the w_j^i for each kernel.
    def _std_ji_rough(self, tj: float, ti_s: np.ndarray | list) -> np.ndarray:
        std_ji = np.sqrt((self.eta**2) * ((tj-ti_s[:-1])**(2*self.H) - (tj-ti_s[1:])**(2*self.H)) / (2*self.H))
        return std_ji

    def _std_ji_path_dependent(self, tj: float, ti_s: np.ndarray | list) -> np.ndarray:
        if self.H == 0:
            std_ji = np.sqrt((self.eta**2) * (np.log(tj-ti_s[:-1]+self.eps) - np.log(tj-ti_s[1:]+self.eps)))
        else:
            std_ji = np.sqrt((self.eta**2) * ((tj-ti_s[:-1]+self.eps)**(2*self.H) - (tj-ti_s[1:]+self.eps)**(2*self.H)) / (2*self.H))
        return std_ji

    def _std_ji_one_factor(self, tj: float, ti_s: np.ndarray | list) -> np.ndarray:
        return self.eta * self.eps ** self.H * np.sqrt((np.exp((2*self.H-1)*(tj-ti_s[:-1])/self.eps) - np.exp((2*self.H-1)*(tj-ti_s[1:])/self.eps)) / (2*self.H-1))
    
    def _std_ji_two_factor(self, tj: float, ti_s: np.ndarray | list) -> np.ndarray:
        acst = (self.eta1**2) * (self.eps1**(2*self.H1)) / (2*self.H1-1)
        bcst = (self.eta2**2) * (self.eps2**(2*self.H2)) / (2*self.H2-1)
        gamma = (1/self.eps1)*(self.H1-0.5)+(1/self.eps2)*(self.H2-0.5)
        if gamma == 0:
            ccst = 2*self.eta1*self.eta2*(self.eps1**(self.H1-0.5))*(self.eps2**(self.H2-0.5))
        else:
            ccst = 2*self.eta1*self.eta2*(self.eps1**(self.H1-0.5))*(self.eps2**(self.H2-0.5)) / gamma

        a_ji = acst * (np.exp((2*self.H1-1)*(tj-ti_s[:-1])/self.eps1) - np.exp((2*self.H1-1)*(tj-ti_s[1:])/self.eps1))
        b_ji = bcst * (np.exp((2*self.H2-1)*(tj-ti_s[:-1])/self.eps2) - np.exp((2*self.H2-1)*(tj-ti_s[1:])/self.eps2))
        if gamma == 0:
            c_ji = ccst*(ti_s[1:]-ti_s[:-1])
        else:
            c_ji = ccst*(np.exp(gamma * (tj-ti_s[:-1])) - np.exp(gamma * (tj-ti_s[1:])))

        return np.sqrt(a_ji+b_ji+c_ji)
    
    def _R_bar_rough(self, T: float, n_terms: int=100):
        alpha = - self.a * self.eta**2 * gamma(2 * self.H)
        return (1 / (2*self.H)) * sum(((-1)**k) * (alpha**(k+1)) * (T**(2*self.H*(k+1))) / ((k+1) * gamma(2*self.H*(k+1))) for k in range(n_terms))
    
    def _laplace_k_tilde_path_dependent(self, u):
        if u == 0:
            return self.a*(self.eta**2)*(self.eps**(2*self.H)) / (2*self.H) if self.H > 0 else mp.mpf('+inf')
        return -self.a * (self.eta**2) * mp.exp(u * self.eps) * (1 / (u**(2*self.H))) * mp.gammainc(2*self.H, a=u*self.eps)
    
    def _calculate_resolvent_path_dependent(self, tau):
        resolvent = []
        for t in range(tau):
            resolvent.append(mp.invertlaplace(lambda u: 1 / (1 + (1 / self._laplace_k_tilde_path_dependent(u))), t, method='cohen'))
        return np.array(resolvent)

    def _R_bar_path_dependent(self, t):
        t_wth_zero = np.concatenate([[1e-20], t[1:]], axis=0)
        resolvent = self._calculate_resolvent_path_dependent(t_wth_zero)
        integrale_resolvent = cumulative_trapezoid(resolvent, t_wth_zero)
        integrale_resolvent = np.concatenate([[0], integrale_resolvent], axis=0)
        integrale_resolvent = np.array([float(el) for el in integrale_resolvent])
        return integrale_resolvent
    
    def _R_bar_one_factor(self, t):
        cst = -self.a * (self.eta ** 2) * (self.eps ** (2*self.H-1))
        mul = -(2*self.H-1) / self.eps
        return (cst / (cst + mul)) * (1 - np.exp(-(cst + mul) * t))
    
    def _calculate_laplace_transform_exp(self, u, cst, mul):
        if cst == 0:
            return 0
        if mp.re(u) <= mul:
            return cst*mp.mpf('+inf')
        return cst / (u - mul)

    def _laplace_k_tilde_two_factor(self, u, cst1, cst2, cst3, mul1, mul2, mul3):
        laplace_k_tilde1 = self._calculate_laplace_transform_exp(u, cst1, mul1)
        laplace_k_tilde2 = self._calculate_laplace_transform_exp(u, cst2, mul2)
        laplace_k_tilde3 = self._calculate_laplace_transform_exp(u, cst3, mul3)
        return laplace_k_tilde1 + laplace_k_tilde2 + laplace_k_tilde3

    def _calculate_resolvent_two_factor(self, tau):
        cst1 = -self.a * (self.eta1**2) * (self.eps1 ** (2*self.H1-1))
        cst2 = -2*self.a*self.eta1*self.eta2*(self.eps1 ** (self.H1-0.5))*(self.eps2 ** (self.H2-0.5))
        cst3 = -self.a * (self.eta2**2) * (self.eps2 ** (2*self.H2-1))
        mul1 = (2*self.H1-1) / self.eps1
        mul2 = (1/self.eps1)*(self.H1-0.5)+(1/self.eps2)*(self.H2-0.5)
        mul3 = (2*self.H2-1) / self.eps2
        resolvent = []
        for t in range(tau):
            # With 'talbot', the resolvent diverges to infinity as t gets closer to 0.
            # 'cohen' is more appropriate for dealing with such singularities.
            resolvent.append(mp.invertlaplace(lambda u: 1 / (1 + (1 / self.laplace_k_tilde_two_factor(u, cst1, cst2, cst3, mul1, mul2, mul3))), t, method='cohen'))
        return np.array(resolvent)

    def _R_bar_two_factor(self, t):
        t_wth_zero = np.concatenate([[1e-20], t[1:]], axis=0)
        resolvent = self._calculate_resolvent_two_factor(t_wth_zero)
        integrale_resolvent = cumulative_trapezoid(resolvent, t_wth_zero)
        integrale_resolvent = np.concatenate([[0], integrale_resolvent], axis=0)
        integrale_resolvent = np.array([float(el) for el in integrale_resolvent])
        return integrale_resolvent

    # TODO: looks like we are recomputing the w_j^i each time..?
    def generate_paths(self, n_steps: int, length: int = 1, n_sims: int = 1, kernel: KernelFlavour=KernelFlavour.ROUGH):
        # Uncorrelated brownians
        w1, w2 = (
            np.random.normal(0, 1, (n_steps * length, n_sims)),
            np.random.normal(0, 1, (n_steps * length, n_sims)),
        )
        tt = np.linspace(0, n_steps * length, n_steps * length + 1)

        # Initiate rough Heston Z process and quadratic form instantaneous variance V
        dt = tt[1] - tt[0] # Uniform grid
        N_sims = w1.shape[1]
        Z = np.zeros(N_sims).reshape(1,-1)
        V = np.zeros(N_sims).reshape(1,-1)
        Z[0] = self.fvc
        V[0] = self.a*(self.fvc-self.b)**2+self.c


        for j in range(n_steps * length):
            tj = tt[j+1]
            ti_s = tt[:j+2]

            if kernel in self.kernel_std:
                std_ji = self.kernel_std[kernel](tj, ti_s)
            else:
                warnings.warn("Unrecognized kernel type. Please provide a valid kernel type from KernelFlavour enum.", UserWarning)
                raise ValueError("Invalid kernel type provided.")
            
            Z_temp = self.fvc + np.sum(std_ji.reshape(-1, 1)*w1[:j+1]*np.sqrt(V[:j+1]), axis=0)
            V_temp = self.a*((Z_temp-self.b)**2)+self.c
            Z = np.append(Z, Z_temp.reshape(1, -1), axis=0)
            V = np.append(V, V_temp.reshape(1, -1), axis=0)

        # Correlate the brownians with rho
        correlated_brownian = self.rho * w1 + np.sqrt(1 - self.rho**2) * w2
        logSt_increment = -0.5 * dt * V[:-1, :] + np.sqrt(dt) * np.multiply(np.sqrt(V[:-1, :]), correlated_brownian)
        logSt_increment = np.concatenate([np.full((1, n_sims), np.log(self.initial_price)), logSt_increment], axis=0)
        logSt = np.cumsum(logSt_increment, axis=0)
        St = np.exp(logSt)

        self.grid = tt # Needed to match grids with VIX levels computations
        self.w1 = w1

        return St, V
    
    def compute_vix(self, t, tau, delta, V, n_steps:int, n_sims:int, kernel: KernelFlavour=KernelFlavour.ROUGH):
        tt = self.grid

        if kernel==KernelFlavour.ROUGH:
            int_res = self._R_bar_rough(T=tau)
        elif kernel==KernelFlavour.PATH_DEPENDENT:
            int_res = self._R_bar_path_dependent(t=tau)
        elif kernel==KernelFlavour.ONE_FACTOR:
            int_res = self._R_bar_one_factor(t=tau)
        elif kernel == KernelFlavour.TWO_FACTOR:
            int_res = self._R_bar_two_factor(t=tau)
        else:
            warnings.warn("Unrecognized kernel type. Please provide a valid kernel type from KernelFlavour enum.", UserWarning)
            raise ValueError("Invalid kernel type provided.")
        
        int_res = 1 - np.flip(int_res)
        int_res = int_res.reshape((-1, 1))

        vix=[]
        for time_t in range(t): # Nested loop as we integrate of the forward variance at each time step to get VIX levels
            j = int(n_steps * time_t)
            f = []
            for time_tau in range(tau):
                k = int(n_steps*time_tau)
                tj = tt[j+k+1]
                ti_s = tt[:j+2]

                if kernel in self.kernel_std:
                    std_ji = self.kernel_std[kernel](tj, ti_s)
                else:
                    warnings.warn("Unrecognized kernel type. Please provide a valid kernel type from KernelFlavour enum.", UserWarning)
                    raise ValueError("Invalid kernel type provided.")
                
                g = self.fvc + np.sum(std_ji.reshape(-1, 1)*self.w1[:j+1]*np.sqrt(V[:j+1]), axis=0)
                f.append((self.a * (g - self.b)**2 + self.c).reshape((1, -1)))
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


    def generate_paths_old(self, n_steps: int, length: int, n_sims: int = 1):
        # Uncorrelated brownians
        w1, w2 = (
            np.random.normal(0, 1, (n_steps * length, n_sims)),
            np.random.normal(0, 1, (n_steps * length, n_sims)),
        )

        # Initiate rough Heston Z process and quadratic form instantaneous variance V
        Zt = np.full((1, n_sims), self.fvc)
        Vt = np.full((1, n_sims), self.a*(self.fvc - self.b)**2 + self.c)
        Ztj = Zt[0, :]
        Vtj = Vt[0, :]

        tt = np.linspace(0, n_steps* length, n_steps * length + 1)
        dt = 1 / n_steps

        w_tilde = np.sqrt(
            (dt ** (2 * self.H) * ((tt[1:] ** (2 * self.H)) - (tt[:-1] ** (2 * self.H))))
            / (2 * self.H)
        )

        for j in range(n_steps * length):
            w_tilde_diff = np.flip(w_tilde[: j + 1], axis=0) - np.append(
                np.flip(w_tilde[:j], axis=0), [0]
            )
            w_tilde_diff = w_tilde_diff.reshape(-1, 1)
            w_tilde_diff = np.concatenate([w_tilde_diff] * n_sims, axis=1)
            z_increment = np.multiply(w_tilde_diff, w1[: j + 1, :]).sum(axis=0)
            z_increment = np.multiply(np.sqrt(Vtj), z_increment)
            Ztj_next = Ztj + z_increment
            Vtj_next = self.a * (Ztj_next - self.b) ** 2 + self.c
            Ztj_next = Ztj_next.reshape(1, -1)
            Vtj_next = Vtj_next.reshape(1, -1)
            Zt = np.concatenate([Zt, Ztj_next], axis=0)
            Vt = np.concatenate([Vt, Vtj_next], axis=0)
            Ztj = Ztj_next
            Vtj = Vtj_next

        # Correlate the brownians with rho
        correlated_brownian = self.rho * w1 + np.sqrt(1 - self.rho**2) * w2
        correlated_brownian = np.concatenate(
            [np.zeros((1, n_sims)), correlated_brownian], axis=0
        )

        logSt_increment = -0.5 * dt * Vt + np.sqrt(dt) * np.multiply(
            np.sqrt(Vt), correlated_brownian
        )
        logSt_increment[0] = np.full((1, n_sims), np.log(self.initial_price))
        logSt = np.cumsum(logSt_increment, axis=0)
        prices = np.exp(logSt)

        # TODO
        self.grid = tt # Needed to match grids with VIX levels computations

        return prices, Vt

    # def resolvent(self, t, kernel: KernelFlavour):
    #     """
    #     For the quadratic Volterra Heston models, we are interested in the resolvent of -aK^2.
    #     For the rough kernel: with c=-a \eta^2 \Gamma(2H), alpha=2H, R(t)=ct^(alpha-1) E_{alpha, alpha}(-ct^(alpha)),
    #     For the path-dependent kernel: mutatis mutandis t, t+eps,
    #     For the one-factor kernel: c=-a \eta^2 \epsilon^(2H-1), \lambda=(1_2H) / \epsilon, R(t)=c e^{-(\lambda+c)t}.
    #     """
    #     if kernel == KernelFlavour.ROUGH:
    #         c = -self.a * self.eta**2 * gamma(2 * self.H)
    #         return (
    #             c * t ** (2 * self.H - 1)
    #             * mittag_leffler(
    #                 alpha=2 * self.H, beta=2 * self.H, z=-c * t ** (2 * self.H)
    #             )
    #         )
    #     elif kernel == KernelFlavour.PATH_DEPENDENT:
    #         c = -self.a * self.eta**2 * gamma(2 * self.H)
    #         return (
    #             c * (t + self.eps) ** (2 * self.H - 1)
    #             * mittag_leffler(
    #                 alpha=2 * self.H,
    #                 beta=2 * self.H,
    #                 z=-c * (t + self.eps) ** (2 * self.H),
    #             )
    #         )
    #     elif kernel == KernelFlavour.ONE_FACTOR:
    #         c = -self.a * self.eta**2 * self.eps ** (2 * self.H - 1)
    #         lamb = (1 - 2 * self.H) / self.eps
    #         return c * np.exp(-(lamb + c) * t)

    def generate_VIX_levels(
        self,
        maturity: float = 1 / 12,
        delta: float = 1 / 12,
        kernel: KernelFlavour = KernelFlavour.ROUGH,
        n_steps: int = 100,
    ):
        """
        We approximate an integrand involving the resolvent to generate coherent VIX data.

        \mbox{VIX}_T^2 = \frac{100^2}{\Delta} \int_T^{T+\Delta} \left[1-\bar{R}(\Delta - u)\right]V_u du
        """
        integral = 0
        # We first need an instantaneous variance trajectory, beware first value is 0
        _, variance = self.generate_paths(n_steps=n_steps + 1, length=1)

        for index, u in enumerate(
            np.linspace(maturity, maturity + delta, n_steps + 1)[:-1]
        ):  # Issues at end, remove last value
            # Then, compute the mean resolvent
            times = np.linspace(0, maturity + delta - u, 100)[
                1:
            ]  # Divide the interval [0, t] into 100 points, set 0->0
            resolvent_values = np.array(
                [self.resolvent(t=time, kernel=kernel) for time in times]
            )
            resolvent_bar = trapz(resolvent_values, times)

            integral += (1 - resolvent_bar) * variance[index + 1]

        return (100**2) / delta * integral * (1/n_steps)
