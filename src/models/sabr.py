import numpy as np
from pysabr import Hagan2002LognormalSABR

from .base_model import BaseModel


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
        alpha, rho, vvol = sabr_lognormal.fit(strikes, vols)
        self.alpha, self.rho, self.vvol = alpha, rho, vvol

    def set_parameters(self, alpha: float=0.25, rho: float=-0.6, vvol: float=2.5):
        self.alpha, self.rho, self.vvol = alpha, rho, vvol

    def generate_paths(self, n_steps, length):
        # Discretization grid
        dt = length / n_steps

        dw = np.sqrt(dt)*np.random.normal(0, 1, size=n_steps)

        dz = np.zeros_like(dw)
        dz[0] = dw[0]
        for i in range(1, n_steps):
            # Update the correlation
            dz[i] = self.rho * dz[i-1] + np.sqrt(1 - self.rho**2) * dw[i]
        
        F = np.zeros(n_steps)
        vol = np.zeros_like(F) 
    
        F[0] = self.initial_price
        vol[0] = self.alpha

        for t in range(1, n_steps):
            vol[t] = vol[t - 1] * (1 + self.vvol * dz[t])

            F[t] = F[t - 1] + vol[t] * dw[t]

        return F

