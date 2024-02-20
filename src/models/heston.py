from src.models.base_model import BaseModel


class Heston(BaseModel):
    def __init__(self, initial_price: float = 100):
        """
        The Heston model is a stochastic volatility model that assumes a geometric brownian motion dynamic
        for the stock price and a CIR process for the instantaneous variance:
        
        dS_t &= \mu S_t \, dt + \sqrt{V_t}S_t \, dW_t \\
        dV_t &= \kappa (\theta - v_t) \, dt + \sigma \sqrt{V_t} \, dB_t
        \langle dW_t, dB_t \rangle = \rho dt
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
        rho: float = -0.6,
    ):
        """
        Set of coherent dummy parameters to play around without market data.
        """
        (
            self.vol_init,
            self.mu,
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
            self.alpha,
        ) = vol_init, mu, kappa, theta, sigma, rho

    
