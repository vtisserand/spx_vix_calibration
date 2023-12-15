from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
from scipy.optimize import curve_fit


class FitType:
    NONE = "none"
    EXP = "exp"
    POWER = "power"


def exponential_fit(x, a, b, c):
    return -a * np.exp(-b * x) + c


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

        mean_returns = np.mean(returns[: -self.lag])
        std_returns = np.std(returns[: -self.lag])

        # Calculate kurtosis
        numerator = np.mean((returns[self.lag :] - mean_returns) ** 4)
        denominator = std_returns**4

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
    def __init__(
        self, prices: np.ndarray | list, lag: int = 10, threshold: float = 0.5
    ):
        self.prices = prices
        self.lag = lag
        self.threshold = threshold

    def compute_cross_correlation(self):
        returns = np.diff(np.log(self.prices))
        absolute_returns = np.abs(returns)

        # Calculate correlation between squared absolute returns and returns at lag
        squared_absolute_returns = absolute_returns**2
        correlation = signal.correlate(
            squared_absolute_returns[: -self.lag], returns[self.lag :], mode="same"
        )

        # Trim to actually get cross-correlation
        correlation = correlation[len(correlation) // 2 + self.lag :]

        normalization = np.correlate(
            squared_absolute_returns, squared_absolute_returns
        )  # Normalize

        leverage_effect_corr = correlation / normalization
        return leverage_effect_corr

    def plot(
        self,
        window: int = 200,
        fit_type: FitType = FitType.NONE,
        show_confidence_bounds: bool = False,
        return_obj: bool=False,
    ):
        fig, ax = plt.subplots()

        corr = self.compute_cross_correlation()[:window]
        time_axis = np.arange(len(corr))

        # Plot the optional fit
        if fit_type == FitType.EXP:
            fit_coefficients, _ = curve_fit(exponential_fit, np.arange(len(corr)), corr)

            # Generate the fitted curve
            fit_curve = exponential_fit(time_axis, *fit_coefficients)

            equation_str = f"Exponential Fit: $-{fit_coefficients[0]:.4f} * \exp{{(-x/{fit_coefficients[1]:.4f})}}$"
            ax.plot(
                time_axis, fit_curve, linestyle="--", color="red", label=equation_str
            )

        if show_confidence_bounds:
            pass

        # Plot the correlation values
        ax.plot(time_axis, corr, label="Correlation")
        ax.set_title(
            "Cross-Correlation of squared absolute returns and returns\n Leverage effect"
        )
        ax.legend()
        ax.grid(True)
        if return_obj:
            return fig
        else:
            fig.show()

    def is_verified(self):
        pass


class ZumbachEffect(StylizedFact):
    def __init__(self, prices: np.ndarray | list, lag: int = 10, threshold: float = 0.5):
        self.prices = prices
        self.lag = lag
        self.threshold = threshold

    def compute_cross_correlation(self):
        returns = np.diff(np.log(self.prices))
        ts = pd.Series(returns)
        vols = ts.rolling(window=15).std().to_numpy() * (258**0.5)
        square_returns = np.square(returns)

        correlation = signal.correlate(
            square_returns[: -self.lag],
            np.where(np.isnan(vols), 0, vols)[self.lag :],
            mode="same",
        )

        # Trim to actually get cross-correlation
        correlation = correlation[len(correlation) // 2 + self.lag :]

        normalization = np.correlate(
            square_returns, np.where(np.isnan(vols), 0, vols)
        )  # Normalize

        zumbach_effect_corr = correlation / normalization
        return zumbach_effect_corr

    def plot(self, window: int = 200, return_obj: bool=False):
        fig, ax = plt.subplots()

        corr = self.compute_cross_correlation()[:window]
        time_axis = np.arange(len(corr))

        # Plot the correlation values
        ax.plot(time_axis, corr, label="Correlation")
        ax.set_title("Cross-Correlation of\n Zumbach effect")
        ax.legend()
        ax.grid(True)
        if return_obj:
            return fig
        else:
            fig.show()

    def is_verified(self):
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

def generate_pdf(prices, name: str='combined_plots'):
    pdf_filename = 'combined_plots.pdf'

    with PdfPages(pdf_filename) as pdf:
        leverage_effect = LeverageEffect(prices)
        zumbach_effect = ZumbachEffect(prices)

        leverage_plot = leverage_effect.plot(fit_type=FitType.EXP, return_obj=True)
        zumbach_plot = zumbach_effect.plot(return_obj=True)

        leverage_plot.savefig(pdf, format='pdf')
        zumbach_plot.savefig(pdf, format='pdf')



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
