import argparse
import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from rich.logging import RichHandler
from scipy import signal
from scipy.optimize import curve_fit

from utils import (
    get_vol_estimates,
    plot_crosscorrelations,
    extract_data_from_axes,
)
from config import NB_SAMPLE_PER_HOUR, NB_TRADING_HOURS_PER_DAY

LOGGER = logging.getLogger("rich")


class FitType:
    NONE = None
    EXP = "exp"
    POWER = "power"


def exponential_fit(x, a, b, c):
    return -a * np.exp(-b * x) + c


def power_fit(x, a, b, c):
    return a * x**b + c


class StylizedFact(ABC):
    def __init__(self, prices: np.ndarray | list, daily: bool = True, vols: np.ndarray | list = None) -> None:
        self.prices = prices
        if daily:
            self.daily_prices = prices
            self.vols = vols
        else:
            self.daily_prices = prices[::int(NB_SAMPLE_PER_HOUR*NB_TRADING_HOURS_PER_DAY)]
            self.vols = get_vol_estimates(self.prices, nb_daily_samples=int(NB_SAMPLE_PER_HOUR*NB_TRADING_HOURS_PER_DAY))
         
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
    def __init__(self,
                 prices: ndarray | list,
                 daily: bool = True,
                 threshold: float = 0.5,
                 lag: int = 15) -> None:
        super().__init__(prices, daily=daily)
        self.threshold = threshold
        self.lag = lag

    def is_verified(self, prices):
        returns = np.diff(np.log(prices))
        autocorrelation = np.correlate(
            returns[: -self.lag], returns[self.lag :], mode="full"
        )
        autocorrelation /= np.max(autocorrelation)  # Normalize

        returns_autocorr = autocorrelation[len(autocorrelation) // 2]
        return returns_autocorr > self.threshold

    def plot(self, return_obj: bool = False):
        returns = np.diff(np.log(self.daily_prices))
        x = returns
        y = returns
        fig = plot_crosscorrelations(x, y, nlags=self.lag, alpha=0.05)

        if return_obj:
            return fig
        else:
            fig.show()


class HeavyTailsKurtosis(StylizedFact):
    def __init__(self,
                 prices: ndarray | list,
                 daily: bool = True,
                 period: int = 30,
                 threshold: float = 0.5) -> None:
        super().__init__(prices, daily=daily)
        self.threshold = threshold
        self.period = period

    def is_verified(self):
        returns = np.diff(np.log(self.daily_prices))
        kurtosis = self.calculate_kurtosis(returns)

        return kurtosis > self.threshold

    @staticmethod
    def calculate_kurtosis(returns):
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)

        # Calculate empirical kurtosis
        numerator = np.mean((returns - mean_returns) ** 4)
        denominator = std_returns**4
        excess_kurtosis = numerator / denominator - 3

        return excess_kurtosis

    def compute_list_kurtosis(self, period=30):
        list_kurtosis = []
        for i in range(1, period + 1):
            returns = np.diff(np.log(self.daily_prices[::i]))
            kurtosis = self.calculate_kurtosis(returns)
            list_kurtosis.append(kurtosis)

        return list_kurtosis

    def plot(self, return_obj: bool = False):
        list_kurtosis = self.compute_list_kurtosis(period=self.period)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        ax.plot(
            list(range(1, self.period + 1)),
            list_kurtosis,
            color="royalblue",
            label="Excess Kurtosis",
        )
        ax.set_xlabel("Period")
        ax.axhline(y=0, color="crimson", linestyle="--")
        ax.set_title("Excess kurtosis of returns")
        ax.grid(which="both", linestyle="--", linewidth=0.5)

        if return_obj:
            return fig
        else:
            fig.show()


class GainLossSkew(StylizedFact):
    def __init__(self,
                 prices: ndarray | list,
                 daily: bool = True,
                 period: int=30,
                 threshold: float=0.5) -> None:
        super().__init__(prices, daily=daily)
        self.threshold = threshold
        self.period = period

    def is_verified(self):
        returns = np.diff(np.log(self.daily_prices))
        skewness = self.calculate_skewness(returns)

        return np.abs(skewness) > self.threshold

    @staticmethod
    def calculate_skewness(returns):
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)

        # Calculate empirical skewness
        numerator = np.mean((returns - mean_returns) ** 3)
        denominator = std_returns**3
        skewness = numerator / denominator

        return skewness

    def compute_list_skewness(self, period=30):
        list_skewness = []
        for i in range(1, period + 1):
            returns = np.diff(np.log(self.daily_prices[::i]))
            skewness = self.calculate_skewness(returns)
            list_skewness.append(skewness)

        return list_skewness

    def plot(self, return_obj: bool = False):
        list_skewness = self.compute_list_skewness(period=self.period)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(
            list(range(1, self.period + 1)),
            list_skewness,
            color="royalblue",
            label="Skewness",
        )
        ax.set_xlabel("Period")
        ax.axhline(y=0, color="crimson", linestyle="--")
        ax.set_title("Skewness of returns")
        ax.grid(which="both", linestyle="--", linewidth=0.5)

        if return_obj:
            return fig
        else:
            fig.show()


class VolatilityClustering(StylizedFact):
    def __init__(self,
                 prices: ndarray | list,
                 daily: bool = True,
                 lag: int=500,
                 threshold: float=0.5) -> None:
        super().__init__(prices, daily=daily)
        self.threshold = threshold
        self.lag = lag

    def plot(
        self, fit_type: FitType = FitType.NONE, alpha=0.05, return_obj: bool = False
    ):
        returns = np.diff(np.log(self.daily_prices))
        x = np.abs(returns)
        fig = plot_crosscorrelations(x, x, nlags=self.lag, alpha=alpha, fit_type=fit_type)

        if return_obj:
            return fig
        else:
            fig.show()

    def is_verified(self, prices, adjust_denominator=False):
        returns = np.diff(np.log(prices))
        x = np.abs(returns)
        y = x
        sd_x = x.std(ddof=0)
        sd_y = y.std(ddof=0)
        x -= x.mean()
        y -= y.mean()

        arr_corr_left = x[: -self.lag]
        arr_corr_right = y[self.lag :]
        n = len(arr_corr_left) + self.lag
        r = (1 / (n - self.lag * adjust_denominator)) * np.correlate(
            arr_corr_left, arr_corr_right
        )[0]
        ccf = r / (sd_x * sd_y)

        return ccf > self.threshold


class LeverageEffect(StylizedFact):
    def __init__(self,
                 prices: ndarray | list,
                 daily: bool = True,
                 lag: int=10,
                 threshold: float=0.5) -> None:
        super().__init__(prices, daily=daily)
        self.threshold = threshold
        self.lag = lag

    def compute_cross_correlation(self):
        returns = np.diff(np.log(self.daily_prices))
        absolute_returns = np.abs(returns)

        # Calculate correlation between squared absolute returns and returns at lag
        squared_absolute_returns = absolute_returns**2
        correlation = signal.correlate(
            squared_absolute_returns[: -self.lag], returns[self.lag :], mode="same"
        )

        normalization = np.correlate(
            squared_absolute_returns, squared_absolute_returns
        )  # Normalize

        leverage_effect_corr = correlation / normalization
        return leverage_effect_corr

    def plot(
        self,
        window: int = 200,
        fit_type: FitType = FitType.NONE,
        tra: bool = False,
        show_confidence_bounds: bool = False,
        return_obj: bool = False,
    ):
        fig, ax = plt.subplots()

        correlation = self.compute_cross_correlation()

        # Trim to actually get cross-correlation
        corr = correlation[len(correlation) // 2 + self.lag :][:window]
        time_axis = np.arange(len(corr))

        # Plot the optional fit
        if fit_type == FitType.EXP:
            fit_coefficients, _ = curve_fit(exponential_fit, np.arange(len(corr)), corr)

            # Generate the fitted curve
            fit_curve = exponential_fit(time_axis, *fit_coefficients)

            equation_str = f"Exponential Fit: $-{fit_coefficients[0]:.4f} * \exp{{(-x/{fit_coefficients[1]:.4f})}}$"
            ax.plot(
                time_axis,
                fit_curve,
                linestyle="--",
                color="crimson",
                label=equation_str,
            )

        if show_confidence_bounds:
            pass

        if tra:
            corr_tra = correlation[: len(correlation) // 2 + self.lag - 1][::-1][
                :window
            ]
            ax.scatter(
                np.arange(len(corr_tra)),
                corr_tra,
                label="Correlation time reversal",
                color="seagreen",
                s=10,
                alpha=0.4,
            )

        # Plot the correlation values
        ax.scatter(time_axis, corr, label="Correlation", color="royalblue", s=10)
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
    def __init__(self,
                 prices: ndarray | list,
                 daily: bool = True,
                 vols: np.ndarray | list = None,
                 lag: int=10,
                 threshold: float=0.5) -> None:
        super().__init__(prices, daily=daily, vols=vols)
        self.threshold = threshold
        self.lag = lag

    def compute_cross_correlation(self):
        returns = np.diff(np.log(self.daily_prices))
        square_returns = np.square(returns)

        correlation = signal.correlate(
            square_returns[: -self.lag],
            np.where(np.isnan(self.vols), 0, self.vols)[self.lag :],
            mode="same",
        )

        if len(self.vols) > len(square_returns):
            self.vols = self.vols[1:]

        normalization = np.correlate(
            square_returns, np.where(np.isnan(self.vols), 0, self.vols)
        )  # Normalize

        zumbach_effect_corr = correlation / normalization
        return zumbach_effect_corr

    def plot(self, window: int = 200, tra: bool = False, return_obj: bool = False):
        fig, ax = plt.subplots()

        correlation = self.compute_cross_correlation()

        # Trim to actually get cross-correlation
        corr = correlation[len(correlation) // 2 + self.lag :][:window]
        time_axis = np.arange(len(corr))

        if tra:
            corr_tra = correlation[: len(correlation) // 2 + self.lag - 1][::-1][
                :window
            ]
            ax.plot(
                np.arange(len(corr_tra)),
                corr_tra,
                label="Correlation time reversal",
                color="green",
                alpha=0.4,
            )

        # Plot the correlation values
        ax.plot(time_axis, corr, label="Correlation", color="blue")
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


def get_desc_plots(prices, daily=True, vols=None):

    if daily:
        daily_prices = prices
    else:
        vols = get_vol_estimates(
            prices, nb_daily_samples=int(NB_SAMPLE_PER_HOUR * NB_TRADING_HOURS_PER_DAY)
        )
        daily_prices = prices[::int(NB_SAMPLE_PER_HOUR * NB_TRADING_HOURS_PER_DAY)]

    daily_returns = np.diff(np.log(daily_prices))

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]})

    # Prices plot on the first y-axis
    time_axis = np.arange(len(daily_prices))
    axs[0].plot(time_axis, daily_prices, color="b")
    axs[0].set_ylabel("Prices", color="b")
    axs[0].tick_params("y", colors="b")

    # Volatility plot on the second y-axis
    ax2 = axs[0].twinx()
    ax2.plot(time_axis[len(daily_prices) - len(vols):], vols, color="r")
    ax2.set_ylabel("Volatility", color="r")
    ax2.tick_params("y", colors="r")
    axs[0].grid(True)

    # Title and legend
    axs[0].set_title("Descriptive plots of the prices sequence")

    time_axis = np.arange(len(daily_returns))
    axs[1].plot(time_axis, daily_returns, color="b")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Returns", color="b")
    axs[1].tick_params("y", colors="b")
    axs[1].grid(True)

    return fig


def generate_pdf(prices, daily, name: str = "combined_plots", vols: np.ndarray | list = None):
    file_extension = ".pdf"
    pdf_filename = "".join([name, file_extension])

    with PdfPages(pdf_filename) as pdf:
        desc_plot = get_desc_plots(prices, daily=daily, vols=vols)
        LOGGER.info("Descriptive plots generated.")
        return_acf = ReturnsAutocorrelation(prices, daily=daily)
        kurtosis = HeavyTailsKurtosis(prices, daily=daily)
        skewness = GainLossSkew(prices, daily=daily)
        vol_cluster = VolatilityClustering(prices, daily=daily)
        leverage_effect = LeverageEffect(prices, daily=daily)
        zumbach_effect = ZumbachEffect(prices, daily=daily, vols=vols)

        returns_acf_plot = return_acf.plot(return_obj=True)
        LOGGER.info("Returns ACF plot generated.")
        kurtosis_plot = kurtosis.plot(return_obj=True)
        LOGGER.info("Kurtosis plot generated.")
        skewness_plot = skewness.plot(return_obj=True)
        LOGGER.info("Skewness plot generated.")
        vol_cluster_expo_plot = vol_cluster.plot(return_obj=True, fit_type=FitType.EXP)
        LOGGER.info("Volatility clustering plot (exponential fit) generated.")
        vol_cluster_power_plot = vol_cluster.plot(return_obj=True, fit_type=FitType.POWER)
        LOGGER.info("Volatility clustering plot (power fit) generated.")
        leverage_plot = leverage_effect.plot(
            tra=True, fit_type=FitType.EXP, return_obj=True
        )
        LOGGER.info("Leverage effect plot generated.")
        zumbach_plot = zumbach_effect.plot(tra=True, return_obj=True)
        LOGGER.info("Zumbach effect plot generated.")

        desc_plot.savefig(pdf, format="pdf")
        returns_acf_plot.savefig(pdf, format="pdf")
        kurtosis_plot.savefig(pdf, format="pdf")
        skewness_plot.savefig(pdf, format="pdf")
        vol_cluster_expo_plot.savefig(pdf, format="pdf")
        vol_cluster_power_plot.savefig(pdf, format="pdf")
        leverage_plot.savefig(pdf, format="pdf")
        zumbach_plot.savefig(pdf, format="pdf")


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
