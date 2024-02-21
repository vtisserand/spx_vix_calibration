import numpy as np
from scipy.special import gamma 

from src.models.base_model import BaseModel


class rBergomi(BaseModel):
    def __init__(self, initial_price: float = 100):
        """
        The rough Bergomi model...
        """
        super().__init__(initial_price)
        self.set_parameters()

    def set_parameters(
        self,
        vol_init: float = 0.25,
        mu: float = 0.1,
        kappa: float = 1.1,
        theta: float = 0.12,
        sigma: float = 0.80,
        rho: float = -0.7,
    ):
        """
        Set of coherent dummy parameters to play around without market data, from the original rough Heston paper.
        """
        (
            self.vol_init,
            self.mu,
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
        ) = vol_init, mu, kappa, theta, sigma, rho


    def fit(
        self,
        strikes: np.ndarray | list[float],
        prices: np.ndarray | list[float],
        forward_price: float = 100,
        maturity: float = 1,
    ):
        """
        Fits a Heston model on a surface slice by minimizing the MSE between market prices and model prices.
        """
        pass

    def generate_paths(self, n_steps: int, length: int, n_sims: int=1):
        pass
