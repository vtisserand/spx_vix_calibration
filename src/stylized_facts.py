import numpy as np
from abc import ABC, abstractmethod

class StylizedFact(ABC):
    @abstractmethod
    def is_verified(self, prices):
        """
        Check if the stylized fact is verified in the given price trajectory.

        Parameters:
        - prices (list or np.ndarray): Price trajectory.

        Returns:
        - bool: True if the stylized fact is verified, False otherwise.
        """
        pass

class ReturnsAutocorrelation(StylizedFact):
    def __init__(self, lag=1, threshold=0.5):
        self.lag = lag
        self.threshold = threshold

    def is_verified(self, prices):
        returns = np.diff(np.log(prices))
        autocorrelation = np.correlate(returns[:-self.lag], returns[self.lag:], mode='full')
        autocorrelation /= np.max(autocorrelation)  # Normalize

        returns_autocorr = autocorrelation[len(autocorrelation) // 2]
        return returns_autocorr > self.threshold

class HeavyTailsKurtosis(StylizedFact):
    def __init__(self, lag=1, threshold=0.5):
        self.lag = lag
        self.threshold = threshold

    def is_verified(self, prices):
        returns = np.diff(np.log(prices))

        mean_returns = np.mean(returns[:-self.lag])
        std_returns = np.std(returns[:-self.lag])

        # Calculate kurtosis
        numerator = np.mean((returns[self.lag:] - mean_returns) ** 4)
        denominator = std_returns ** 4

        kurtosis = numerator / denominator - 3

        return kurtosis > self.threshold


