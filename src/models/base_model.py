from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    def __init__(self, initial_price, drift=0.0):
        self.initial_price = initial_price
        self.drift = drift

    @abstractmethod
    def generate_trajectory(self, num_steps, time_step):
        pass


class GeometricBrownianMotion(BaseModel):
    def generate_trajectory(self, volatility, num_steps, time_step):
        prices = [self.initial_price]
        price = self.initial_price
        self.volatility = volatility

        for _ in range(num_steps):
            dW = np.random.normal(0, 1) * np.sqrt(time_step)
            price = self._next_price(price, time_step, dW)
            prices.append(price)

        return prices

    def _next_price(self, current_price, time_step, dW):
        return current_price * np.exp(
            (self.drift - 0.5 * self.volatility**2) * time_step + self.volatility * dW
        )
