import numpy as np

from base_model import BaseModel

class FractionalBrownianMotion(BaseModel):
    def __init__(self, initial_price: float, volatility: float, hurst_parameter: float, drift: float=0.0):
        super().__init__(initial_price, drift)
        self.hurst_parameter = hurst_parameter
        self.volatility = volatility

    def generate_trajectory(self, num_steps: int, time_step: float):
        prices = [self.initial_price]
        price = self.initial_price

        for _ in range(num_steps):
            dW = np.random.normal(0, 1) * np.sqrt(time_step)
            dBH = self._fractional_brownian_motion(time_step)
            price = self._next_price(price, time_step, dW, dBH)
            prices.append(price)

        return prices

    def _fractional_brownian_motion(self, time_step):
        # Construct fractional Brownian motion using the Davies-Harte method
        n = 2 ** 12
        t_values = np.arange(0, 1 + 1 / n, 1 / n)
        x_values = np.fft.ifft((t_values ** (self.hurst_parameter - 0.5)) * np.random.normal(0, 1, n+1)).real
        W_H = np.cumsum(x_values) * np.sqrt(time_step)
        return W_H[-1]

    def _next_price(self, current_price, time_step, dW, dBH):
        return current_price * np.exp(
            (self.drift - 0.5 * self.volatility**2) * time_step + self.volatility * (dW + dBH)
        )
