import argparse
import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from rich.logging import RichHandler
from scipy import signal
from scipy.optimize import curve_fit

import arch
from arch.unitroot import ADF
from arch.unitroot import DFGLS
from arch.unitroot import PhillipsPerron
from arch.unitroot import KPSS

from utils import plot_crosscorrelations, extract_data_from_axes



LOGGER = logging.getLogger("rich")

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

    def __init__(self, prices: np.ndarray | list, lag: int=15, threshold: float=0.5):
        self.prices = prices
        self.lag = lag
        self.threshold = threshold

    def is_verified(self, prices):
        returns = np.diff(np.log(prices))
        autocorrelation = np.correlate(
            returns[: -self.lag], returns[self.lag:], mode="full"
        )
        autocorrelation /= np.max(autocorrelation)  # Normalize

        returns_autocorr = autocorrelation[len(autocorrelation) // 2]
        return returns_autocorr > self.threshold
    
    def plot(self, return_obj: bool=False):
        returns = np.diff(np.log(self.prices))
        x = returns
        y = returns
        fig = plot_crosscorrelations(x, y, nlags=self.lag, alpha=0.05)

        if return_obj:
            return fig
        else:
            fig.show()
        

class HeavyTailsKurtosis(StylizedFact):
    def __init__(self, prices: np.ndarray | list, period=30, threshold=0.5):
        self.prices = prices
        self.period = period
        self.threshold = threshold

    def is_verified(self):
        returns = np.diff(np.log(self.prices))
        kurtosis = self.calculate_kurtosis(returns)

        return kurtosis > self.threshold

    @staticmethod
    def calculate_kurtosis(returns):
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)

        # Calculate empirical kurtosis
        numerator = np.mean((returns - mean_returns) ** 4)
        denominator = std_returns ** 4
        excess_kurtosis = numerator / denominator - 3
    
        return excess_kurtosis

    def compute_list_kurtosis(self, period=30):

        list_kurtosis = []
        for i in range(1, period + 1):
            returns = np.diff(np.log(self.prices[::i]))
            kurtosis = self.calculate_kurtosis(returns)
            list_kurtosis.append(kurtosis)

        return list_kurtosis

    def plot(self, return_obj: bool=False):
        list_kurtosis = self.compute_list_kurtosis(period=self.period)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        ax.plot(list(range(1, self.period + 1)), list_kurtosis, color='royalblue', label="Excess Kurtosis")
        ax.set_xlabel("Period")
        ax.axhline(y=0, color="crimson", linestyle="--")
        ax.set_title("Excess kurtosis of returns")
        ax.grid(which='both', linestyle='--', linewidth=0.5)

        if return_obj:
            return fig
        else:
            fig.show()


class GainLossSkew(StylizedFact):
    def __init__(self, prices: np.ndarray | list, period=30, threshold=0.5):
        self.prices = prices
        self.period = period
        self.threshold = threshold

    def is_verified(self):
        returns = np.diff(np.log(self.prices))
        skewness = self.calculate_skewness(returns)

        return np.abs(skewness) > self.threshold

    @staticmethod
    def calculate_skewness(returns):
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)

        # Calculate empirical skewness
        numerator = np.mean((returns - mean_returns) ** 3)
        denominator = std_returns ** 3
        skewness = numerator / denominator
    
        return skewness

    def compute_list_skewness(self, period=30):

        list_skewness = []
        for i in range(1, period + 1):
            returns = np.diff(np.log(self.prices[::i]))
            skewness = self.calculate_skewness(returns)
            list_skewness.append(skewness)

        return list_skewness

    def plot(self, return_obj: bool=False):
        list_skewness = self.compute_list_skewness(period=self.period)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(list(range(1, self.period+1)), list_skewness, color='royalblue', label="Skewness")
        ax.set_xlabel("Period")
        ax.axhline(y=0, color="crimson", linestyle="--")
        ax.set_title("Skewness of returns")
        ax.grid(which='both', linestyle='--', linewidth=0.5)

        if return_obj:
            return fig
        else:
            fig.show()


class VolatilityClustering(StylizedFact):

    def __init__(self, prices: np.ndarray | int, lag: int=100, threshold: float=0.5):
        self.prices = prices
        self.lag = lag
        self.threshold = threshold
    
    def plot(self, fit_type: FitType=FitType.NONE, alpha=0.05, return_obj: bool=False):
        returns = np.diff(np.log(self.prices))
        x = np.abs(returns)
        fig = plot_crosscorrelations(x, x, nlags=self.lag, alpha=alpha)

        ax_acf, ax_pacf = fig.get_axes()[0], fig.get_axes()[1]
        acf, pacf = extract_data_from_axes(ax_acf), extract_data_from_axes(ax_pacf)
        time_axis = np.arange(len(acf))

        # Plot the optional fit
        if fit_type == FitType.POWER:
            acf_fit_coefficients, _ = curve_fit(power_fit, np.arange(len(acf)), acf)

            # Generate the fitted curve
            fit_acf_curve = power_fit(time_axis, *acf_fit_coefficients)

            equation_str = f"Power Fit: $-{acf_fit_coefficients[0]:.4f} * \exp{{(-x/{acf_fit_coefficients[1]:.4f})}}$"
            ax_acf.plot(
                time_axis, fit_acf_curve, linestyle="--", color="crimson", label=equation_str
            )

            pacf_fit_coefficients, _ = curve_fit(power_fit, np.arange(len(pacf)), pacf)

            # Generate the fitted curve
            fit_pacf_curve = power_fit(time_axis, *pacf_fit_coefficients)

            equation_str = f"Power Fit: $-{pacf_fit_coefficients[0]:.4f} * \exp{{(-x/{pacf_fit_coefficients[1]:.4f})}}$"
            ax_pacf.plot(
                time_axis, fit_pacf_curve, linestyle="--", color="crimson", label=equation_str
            )
            
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

        arr_corr_left = x[:-self.lag]
        arr_corr_right = y[self.lag:]
        n = len(arr_corr_left) + self.lag
        r = (1 / (n - self.lag * adjust_denominator)) * np.correlate(
            arr_corr_left, arr_corr_right
        )[0]
        ccf = r / (sd_x * sd_y)

        return ccf > self.threshold





class VolatilityPersistence(StylizedFact):

    def __init__(self, prices: np.ndarray | int, lag: int=100, threshold: float=0.5):
        self.prices = prices
        self.lag = lag
        self.threshold = threshold

    @staticmethod
    def retrieve_test_stats(test, alpha=0.05, decimals=3):
        null_hypothesis = test.null_hypothesis
        p_value = test.pvalue
        if null_hypothesis == 'The process contains a unit root.':
            stationarity = p_value < alpha
        else:
            stationarity = p_value >= alpha
        
        dict_stats = {'Test Statistic': np.round(test.stat, decimals=decimals),
                      'P-Value': np.round(p_value, decimals=decimals),
                      'Lags': test.lags,
                      'Stationarity 5%?': stationarity}
        return dict_stats

    def compute_tests(self, returns, alpha=0.05, decimals=3):

        adf = ADF(returns)
        dfgls = DFGLS(returns)
        pp = PhillipsPerron(returns)
        kpss = KPSS(returns)
        adf_ct = ADF(returns, trend="ct")
        dfgls_ct = DFGLS(returns, trend="ct")
        pp_ct = PhillipsPerron(returns, trend="ct")
        kpss_ct = KPSS(returns, trend="ct")
        
        dict_tests = {'ADF': adf,
                    'DFGLS': dfgls,
                    'Phillips-Perron': pp,
                    'KPSS': kpss,
                    'ADF (constant + time trend)': adf_ct,
                    'DFGLS (constant + time trend)': dfgls_ct,
                    'Phillips-Perron (constant + time trend)': pp_ct,
                    'KPSS (constant + time trend)': kpss_ct}
        
        df_stats = pd.DataFrame()
        for test in dict_tests.keys():
            dict_stats = self.retrieve_test_stats(dict_tests[test], alpha=alpha, decimals=decimals)
            row = pd.DataFrame(data=dict_stats, index=[test])
            df_stats = pd.concat([df_stats, row], axis=0)

        return df_stats

    @staticmethod
    def fit_garch(returns):
        garch = arch.arch_model(returns, vol='garch', p=1, o=0, q=1)
        garch_fitted = garch.fit()

        return garch_fitted
    
    def compute_results(self, returns, alpha=0.05, decimals=3):
        garch = self.fit_garch(returns)
        df_garch = pd.concat([garch.params,
                            garch.std_err,
                            garch.tvalues,
                            garch.pvalues,
                            garch.conf_int()], axis=1)
        df_garch = df_garch.round(decimals=decimals)
        df_garch = df_garch.iloc[1:]
        df_garch = df_garch.reset_index().rename(columns={'index': 'coefs'})
        df_garch.loc[1, 'coefs'] = 'alpha'
        df_garch.loc[2, 'coefs'] = 'beta'
        
        df_stats = self.compute_tests(returns, alpha=alpha, decimals=decimals)
        df_stats = df_stats.reset_index().rename(columns={'index': 'Test Type'})
        
        return df_garch, df_stats

    def plot(self, alpha=0.05, decimals=3, return_obj: bool=False):
        returns = np.diff(np.log(self.prices))
        df_garch, df_stats = self.compute_results(returns, alpha=alpha, decimals=decimals)
        
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 4))
        
        axs[0].axis('tight')
        axs[0].axis('off')
        col_widths = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
        the_table = axs[0].table(cellText=df_garch.values, colLabels=df_garch.columns, loc='center', colWidths=col_widths)
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        axs[0].set_title('GARCH(1, 1) Volatility Model')
        
        axs[1].axis('tight')
        axs[1].axis('off')
        col_widths = [0.4, 0.3, 0.12, 0.12, 0.2]
        the_table = axs[1].table(cellText=df_stats.values, colLabels=df_stats.columns, loc='center', colWidths=col_widths)
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        axs[1].set_title('Return Unit-Root Tests')

        if return_obj:
            return fig
        else:
            fig.show()

    def is_verified(self):
        returns = np.diff(np.log(self.prices))
        garch = self.fit_garch(returns)
        persistence = garch.params[-2:].sum()

        return persistence > self.threshold




































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

        normalization = np.correlate(
            squared_absolute_returns, squared_absolute_returns
        )  # Normalize

        leverage_effect_corr = correlation / normalization
        return leverage_effect_corr

    def plot(
        self,
        window: int = 200,
        fit_type: FitType = FitType.NONE,
        tra: bool=False,
        show_confidence_bounds: bool = False,
        return_obj: bool=False,
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
                time_axis, fit_curve, linestyle="--", color="crimson", label=equation_str
            )

        if show_confidence_bounds:
            pass

        if tra:
            corr_tra = correlation[:len(correlation) // 2 + self.lag -1][::-1][:window]
            ax.scatter(np.arange(len(corr_tra)), corr_tra, label="Correlation time reversal", color='seagreen', s=10, alpha=0.4)

        # Plot the correlation values
        ax.scatter(time_axis, corr, label="Correlation", color='royalblue', s=10)
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

        normalization = np.correlate(
            square_returns, np.where(np.isnan(vols), 0, vols)
        )  # Normalize

        zumbach_effect_corr = correlation / normalization
        return zumbach_effect_corr

    def plot(self, window: int = 200, tra: bool=False, return_obj: bool=False):
        fig, ax = plt.subplots()

        correlation = self.compute_cross_correlation()

        # Trim to actually get cross-correlation
        corr = correlation[len(correlation) // 2 + self.lag :][:window]
        time_axis = np.arange(len(corr))

        if tra:
            corr_tra = correlation[:len(correlation) // 2 + self.lag -1][::-1][:window]
            ax.plot(np.arange(len(corr_tra)), corr_tra, label="Correlation time reversal", color='green', alpha=0.4)

        # Plot the correlation values
        ax.plot(time_axis, corr, label="Correlation", color='blue')
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

def get_desc_plots(prices):
    fig, ax1 = plt.subplots()

    # Prices plot on the first y-axis
    time_axis = np.arange(len(prices))
    ax1.plot(time_axis, prices, color='b')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Prices', color='b')
    ax1.tick_params('y', colors='b')

    # Volatility plot on the second y-axis
    ax2 = ax1.twinx()
    returns = np.diff(np.log(prices))
    ts = pd.Series(returns)
    vols = ts.rolling(window=15).std().to_numpy() * (258**0.5)
    ax2.plot(time_axis[1:], vols, color='r')
    ax2.set_ylabel('Volatility', color='r')
    ax2.tick_params('y', colors='r')

    # Title and legend
    ax1.set_title("Descriptive plots of the prices sequence")
    ax1.grid(True)

    return fig

def generate_pdf(prices, name: str='combined_plots'):
    file_extension = '.pdf'
    pdf_filename = ''.join([name, file_extension])

    with PdfPages(pdf_filename) as pdf:
        desc_plot = get_desc_plots(prices)
        LOGGER.info("Descriptive plots generated.")
        return_acf = ReturnsAutocorrelation(prices)
        kurtosis = HeavyTailsKurtosis(prices)
        skewness = GainLossSkew(prices)
        vol_cluster = VolatilityClustering(prices)
        leverage_effect = LeverageEffect(prices)
        zumbach_effect = ZumbachEffect(prices)

        returns_acf_plot = return_acf.plot(return_obj=True)
        LOGGER.info("Returns ACF plot generated.")
        kurtosis_plot = kurtosis.plot(return_obj=True)
        LOGGER.info("Kurtosis plot generated.")
        skewness_plot = skewness.plot(return_obj=True)
        LOGGER.info("Skewness plot generated.")
        vol_cluster_plot = vol_cluster.plot(return_obj=True)
        LOGGER.info("Volatility clustering plot generated.")
        leverage_plot = leverage_effect.plot(tra=True, fit_type=FitType.EXP, return_obj=True)
        LOGGER.info("Leverage effect plot generated.")
        zumbach_plot = zumbach_effect.plot(tra=True, return_obj=True)
        LOGGER.info("Zumbach effect plot generated.")

        desc_plot.savefig(pdf, format='pdf')
        returns_acf_plot.savefig(pdf, format='pdf')
        kurtosis_plot.savefig(pdf, format='pdf')
        skewness_plot.savefig(pdf, format='pdf')
        vol_cluster_plot.savefig(pdf, format='pdf')
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
