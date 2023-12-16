from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    def __init__(self, initial_price: float=100):
        self.initial_price = initial_price

    @abstractmethod
    def generate_paths(self, num_steps, time_step):
        pass


class GeometricBrownianMotion(BaseModel):
    def generate_paths(self, volatility, drift, num_steps, time_step):
        self.volatility = volatility
        self.drift = drift
        prices = [self.initial_price]
        price = self.initial_price


        for _ in range(num_steps):
            dW = np.random.normal(0, 1) * np.sqrt(time_step)
            price = self._next_price(price, time_step, dW)
            prices.append(price)

        return prices

    def _next_price(self, current_price, time_step, dW):
        return current_price * np.exp(
            (self.drift - 0.5 * self.volatility**2) * time_step + self.volatility * dW
        )
