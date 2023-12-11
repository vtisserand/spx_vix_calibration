import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy.typing as npt

class FitType:
    NONE = 'none'
    EXP = 'exp'
    POWER = 'power'


def exponential_fit(x, a, b, c):
    return a * np.exp(-b * x) + c

def power_fit(x, a, b, c):
    return a * x**b + c

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


# We follow formulas to derive statistics highlighted in: https://arxiv.org/pdf/2311.07738.pdf
# with nice plots when appropriate.


class ReturnsAutocorrelation(StylizedFact):
    def __init__(self, lag=1, threshold=0.5):
        self.lag = lag
        self.threshold = threshold

    def is_verified(self, prices):
        returns = np.diff(np.log(prices))
        autocorrelation = np.correlate(
            returns[: -self.lag], returns[self.lag :], mode="full"
        )
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


class GainLossSkew(StylizedFact):
    def __init__(self, lag=1, threshold=0.5):
        self.lag = lag
        self.threshold = threshold

    def is_verified(self, prices):
        returns = np.diff(np.log(prices))

        mean_returns = np.mean(returns[: -self.lag])
        std_returns = np.std(returns[: -self.lag])

        # Calculate skewness
        numerator = np.mean((returns[self.lag :] - mean_returns) ** 3)
        denominator = std_returns**3

        skewness = numerator / denominator

        return np.abs(skewness) > self.threshold


class VolatilityClustering(StylizedFact):
    def __init__(self, lag=1, threshold=0.5):
        self.lag = lag
        self.threshold = threshold

    def is_verified(self, prices):
        returns = np.diff(np.log(prices))
        absolute_returns = np.abs(returns)

        # Calculate linear autocorrelation of absolute returns
        autocorrelation = np.correlate(
            absolute_returns[: -self.lag], absolute_returns[self.lag :], mode="full"
        )
        autocorrelation /= np.max(autocorrelation)  # Normalize

        volatility_clustering_corr = autocorrelation[len(autocorrelation) // 2]
        return volatility_clustering_corr > self.threshold

    def plot():
        pass


class LeverageEffect(StylizedFact):
    def __init__(self, prices: npt.NDArray | int, lag: int=1, threshold: float=0.5):
        self.prices = prices
        self.lag = lag
        self.threshold = threshold

    def compute_cross_correlation(self, prices):
        returns = np.diff(np.log(prices))
        absolute_returns = np.abs(returns)

        # Calculate correlation between squared absolute returns and returns at lag
        squared_absolute_returns = absolute_returns**2
        correlation = np.correlate(
            squared_absolute_returns[: -self.lag], returns[self.lag :], mode="full"
        )

        # Trim to actually get cross-correlation
        correlation = correlation[len(correlation) // 2 - 1:]

        normalization = np.correlate(squared_absolute_returns, squared_absolute_returns)  # Normalize

        leverage_effect_corr = correlation / normalization 
        return leverage_effect_corr
    
    def plot(self, window: int=200, fit=FitType.NONE):
        corr = self.compute_cross_correlation(self.prices)[:window]
        time_axis = np.arange(len(corr))

        # Plot the correlation values
        plt.plot(time_axis, corr, label='Correlation')
        plt.title('Cross-Correlation of squared absolute returns and returns\n Leverage effect')
        plt.legend()
        plt.grid(True)
        plt.show()

    def is_verified(self, prices):
        pass


class ZumbachEffect(StylizedFact):
    def __init__(self, prices: npt.NDArray | int, lag: int=1, threshold: float=0.5):
        self.prices = prices
        self.lag = lag
        self.threshold = threshold

    def compute_cross_correlation(self, prices):
        returns = np.diff(np.log(prices))
        vols = np.std(returns, ddof=1)

        correlation = np.correlate(
            returns[: -self.lag], vols[self.lag :], mode="full"
        )

        # Trim to actually get cross-correlation
        correlation = correlation[len(correlation) // 2 - 1:]

        leverage_effect_corr = correlation 
        return leverage_effect_corr
    
    def plot(self, window: int=200, fit=FitType.NONE):
        corr = self.compute_cross_correlation(self.prices)[:window]
        time_axis = np.arange(len(corr))

        # Plot the correlation values
        plt.plot(time_axis, corr, label='Correlation')
        plt.title('Cross-Correlation of\n Zumbach effect')
        plt.legend()
        plt.grid(True)
        plt.show()

    def is_verified(self, prices):
        pass


def stylized_fact_pipeline(model_name, model_params, num_steps, time_step, checkers):
    # Generate trajectory using the specified model
    model = globals()[model_name]
    trajectory = model(**model_params).generate_trajectory(num_steps, time_step)

    # Apply checkers
    results = {}
    for checker_name, checker in checkers.items():
        result = checker.is_verified(trajectory)
        results[checker_name] = result

    # Display results
    print(f"\nResults for {model_name} model:")
    for checker_name, result in results.items():
        print(f"{checker_name}: {result}")


# model_params = {
#     'initial_price': initial_price,
#     'volatility': volatility,
#     'hurst_parameter': hurst_parameter,
#     'drift': drift
# }

# # Specify the model and checkers to use
# model_name = 'FractionalBrownianMotion'
# checkers = {
#     'VolatilityPersistence': VolatilityPersistence(lag=1, threshold=0.5),
#     'ZumbachEffect': ZumbachEffect(lag=1, threshold=0.5),
#     'ReturnsAutocorrelation': ReturnsAutocorrelation(lag=1, threshold=0.5),
#     'HeavyTailsKurtosis': HeavyTailsKurtosis(lag=1, threshold=0.5),
#     'GainLossSkew': GainLossSkew(lag=1, threshold=0.5),
#     'VolatilityClustering': VolatilityClustering(lag=1, threshold=0.5),
#     'LeverageEffect': LeverageEffect(lag=1, threshold=0.5),
#     'ZumbachEffectSquaredReturns': ZumbachEffectSquaredReturns(lag=1, threshold=0.5)
# }

# # Execute the pipeline
# stylized_fact_pipeline(model_name, model_params, num_steps, time_step, checkers)
