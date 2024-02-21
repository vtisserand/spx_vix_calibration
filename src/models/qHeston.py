import numpy as np
from scipy.special import gamma
from scipy.integrate import trapz

from src.models.base_model import BaseModel
from src.models.kernels import KernelFlavour, mittag_leffler


class qHeston(BaseModel):
    def __init__(self, initial_price: float = 100):
        """
        The rough Bergomi model...
        """
        super().__init__(initial_price)
        self.set_parameters()

    def set_parameters(
        self,
        vol_init: float=0.25,
        a: float=0.384,
        b: float=0.095,
        c: float=0.0025,
        H: float=0.2,
        eta: float=0.7,
        eps: float=1/52,
        rho: float=-0.6,
        fvc: float=0.35
    ):
        """
        Set of coherent dummy parameters to play around without market data, from the original rough Heston paper.
        """
        (
            self.vol_init,
            self.a,
            self.b,
            self.c,
            self.H,
            self.eta,
            self.eps,
            self.rho,
            self.fvc
        ) = vol_init, a, b, c, H, eta, eps, rho, fvc


    def fit(
        self,
        strikes: np.ndarray | list[float],
        prices: np.ndarray | list[float],
        forward_price: float = 100,
        maturity: float = 1,
    ):
        """
        """
        pass

    def generate_paths(self, n_steps: int, length: int, n_sims: int=1):
        # Uncorrelated brownians
        w1, w2 = np.random.normal(0, 1, (n_steps*length, n_sims)), np.random.normal(0, 1, (n_steps*length, n_sims))

        # Initiate rough Heston Z process and quadratic form instantaneous variance V
        Zt = np.full((1, n_sims), self.fvc)
        Vt = np.zeros(n_sims).reshape(1,-1)
        Ztj = Zt[0, :]
        Vtj = Vt[0, :]

        tt = np.linspace(0,length,n_steps*length+1)
        dt = 1/n_steps

        i = np.arange(n_steps*length)
        i_next = np.arange(1, n_steps*length+1)

        w_tilde = (1 / gamma(self.H + 0.5)) * np.sqrt((dt ** (2 * self.H) * ((i_next ** (2 * self.H)) - (i ** (2 * self.H)))) / (2 * self.H))

        for j in range(n_steps*length):
            w_tilde_diff = np.flip(w_tilde[:j+1], axis=0) - np.append(np.flip(w_tilde[:j], axis=0), [0])
            w_tilde_diff = w_tilde_diff.reshape(-1,1)
            w_tilde_diff = np.concatenate([w_tilde_diff] * n_sims, axis=1)
            z_increment = np.multiply(w_tilde_diff, w1[:j+1, :]).sum(axis=0)
            z_increment = np.multiply(np.sqrt(Vtj), z_increment)
            Ztj_next = Ztj + z_increment
            Vtj_next = self.a * (Ztj_next - self.b) ** 2 + self.c
            Ztj_next = Ztj_next.reshape(1,-1)
            Vtj_next = Vtj_next.reshape(1,-1)
            Zt = np.concatenate([Zt, Ztj_next], axis=0)
            Vt = np.concatenate([Vt, Vtj_next], axis=0)
            Ztj = Ztj_next
            Vtj = Vtj_next

        # Correlate the brownians with rho
        correlated_brownian = self.rho * w1 + np.sqrt(1 - self.rho**2) * w2
        correlated_brownian = np.concatenate([np.zeros((1, n_sims)), correlated_brownian], axis=0)

        logSt_increment = -0.5 * dt * Vt + np.sqrt(dt) * np.multiply(np.sqrt(Vt), correlated_brownian)
        logSt_increment[0] = np.full((1, n_sims), np.log(self.initial_price))
        logSt = np.cumsum(logSt_increment, axis=0)
        prices = np.exp(logSt)

        return prices, Vt
    
    def resolvent(self, t, kernel: KernelFlavour):
        if kernel == KernelFlavour.ROUGH:
            c = -self.a * self.eta**2 * gamma(2*self.H)
            return c * t**(2*self.H-1) * mittag_leffler(alpha=2*self.H, beta=2*self.H, z=-c*t**(2*self.H))

    def generate_VIX_levels(self, maturity: float=1/12, delta: float=1/12, kernel: KernelFlavour=KernelFlavour.ROUGH, n_steps: int=1000,):
        """
        We approximate an integrand involving the resolvent to generate coherent VIX data.

        \mbox{VIX}_T^2 = \frac{100^2}{\Delta} \int_T^{T+\Delta} \left[1-\bar{R}(\Delta - u)\right]V_u du
        """
        integral = 0
        # We first need an instantaneous variance trajectory, beware first value is 0
        _, variance = self.generate_paths(n_steps=n_steps+1, length=1)

        for index, u in enumerate(np.linspace(maturity, maturity+delta, n_steps+1)[:-1]): # Issues at end, remove last value
            # Then, compute the mean resolvent
            times = np.linspace(0, maturity+delta-u, 100)[1:] # Divide the interval [0, t] into 100 points, set 0->0
            resolvent_values = np.array([self.resolvent(t=time, kernel=kernel) for time in times])
            resolvent_bar = trapz(resolvent_values, times)

            integral += (1 - resolvent_bar) * variance[index+1]

        return (100 ** 2) / delta * integral




