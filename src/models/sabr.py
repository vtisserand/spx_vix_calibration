import numpy as np
from pysabr import Hagan2002LognormalSABR

from src.models.base_model import BaseModel


class SABR_Model(BaseModel):
    def __init__(self, initial_price: float = 100):
        super().__init__(initial_price)
        self.set_parameters()

    def fit(
        self,
        strikes: np.ndarray | list[float],
        vols: np.ndarray | list[float],
        forward_price: float = 4000,
        maturity: float = 1,
    ):
        sabr_lognormal = Hagan2002LognormalSABR(f=forward_price, beta=1.0, t=maturity)
        vol_init, rho, alpha = sabr_lognormal.fit(strikes, vols)
        self.vol_init, self.rho, self.alpha = vol_init, rho, alpha

    def set_parameters(
        self, vol_init: float = 0.25, rho: float = -0.6, alpha: float = 2.5
    ):
        self.vol_init, self.rho, self.alpha = vol_init, rho, alpha

    def generate_paths(self, n_steps, length):
        # Discretization grid
        dt = length / n_steps

        dw = np.sqrt(dt) * np.random.normal(0, 1, size=n_steps)

        dz = np.zeros_like(dw)
        dz[0] = dw[0]
        for i in range(1, n_steps):
            # Update the correlation
            dz[i] = self.rho * dz[i - 1] + np.sqrt(1 - self.rho**2) * dw[i]

        F = np.zeros(n_steps)
        vol = np.zeros_like(F)

        F[0] = self.initial_price
        vol[0] = self.vol_init

        for t in range(1, n_steps):
            vol[t] = vol[t - 1] * (1 + self.alpha * dz[t])

            F[t] = F[t - 1] + vol[t] * dw[t]

        return F
