import numpy as np
import scipy
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.compat.numpy import lstsq
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.tsatools import lagmat
from scipy.optimize import curve_fit

from config import NB_DAYS_PER_YEAR

def get_vol_estimates(prices, nb_daily_samples: int):
    """
    For intraday volatility estimates: the usual approach is to
    simulate prices for every 5 minutes interval, keep the daily price and a daily
    vol estimate from the 5 minute samples.
    """
    log_S = np.log(prices)
    realized_variance_i = ((log_S[1:]-log_S[:-1])**2)
    buckets = int(nb_daily_samples) #daily
    realized_variance_daily = []
    for i in range(int((realized_variance_i.shape)[0]/buckets)):
        realized_variance_daily.append(np.sum(realized_variance_i[i*buckets:(i+1)*buckets],axis=0))
    realized_variance_daily = np.array(realized_variance_daily)    
    realized_volatility = np.sqrt(NB_DAYS_PER_YEAR)*np.sqrt(realized_variance_daily)
    return realized_volatility

def calculate_cross_correlation_ols(
    xlags,
    ylags,
    x0,
    y0,
    k,
    adjust_denominator=False,
):
    xparams = lstsq(xlags[k - 1 :, :k], x0[k - 1 :], rcond=None)[0]
    xresiduals = x0[k - 1 :] - xlags[k - 1 :, :k] @ xparams
    xresiduals = xresiduals.reshape(-1)
    yparams = lstsq(ylags[k - 1 :, :k], y0[k - 1 :], rcond=None)[0]
    yresiduals = y0[k - 1 :] - ylags[k - 1 :, :k] @ yparams
    yresiduals = yresiduals.reshape(-1)

    sd_xresiduals = xresiduals.std(ddof=0)
    sd_yresiduals = yresiduals.std(ddof=0)
    xresiduals -= xresiduals.mean()
    yresiduals -= yresiduals.mean()

    pccf = calculate_cross_covariances(
        xresiduals,
        yresiduals,
        sd_xresiduals,
        sd_yresiduals,
        k,
        confint_coef=None,
        adjust_denominator=adjust_denominator,
    )[1]

    return pccf




def cross_correlations_ols(
    x,
    y,
    cross_correlation,
    nlags,
    adjust_denominator=False,
    negative_lags=False,
):
    """
    Calculate partial autocorrelations via OLS.

    """
    xlags, x0 = lagmat(x, nlags, original="sep")
    xlags = add_constant(xlags)
    ylags, y0 = lagmat(y, nlags, original="sep")
    ylags = add_constant(ylags)

    pccf = np.empty(nlags + 1)
    pccf[0] = cross_correlation
    for k in range(1, nlags + 1):
        #print("PCCF lag: {}".format(k))
        ccf = calculate_cross_correlation_ols(
            xlags,
            ylags,
            x0,
            y0,
            k,
            adjust_denominator=adjust_denominator,
        )
        pccf[k] = ccf

    if negative_lags:
        pccf_neg = np.zeros(nlags, np.float64)
        for k in range(1, nlags + 1):
            #print("PCCF lag: {}".format(-k))
            ccf = calculate_cross_correlation_ols(
                ylags,
                xlags,
                y0,
                x0,
                k,
                adjust_denominator=adjust_denominator,
            )
            pccf_neg[k - 1] = ccf
        pccf_neg = np.flip(pccf_neg)
        pccf = np.concatenate([pccf_neg, pccf])

    return pccf




def pacf_yule_walker(r):
    """
    Compute the partial autocorrelations estimates using Yule Walker equations.

    Parameters
    ----------
    r (numpy.ndarray): The autocovariances of the returns for lags 0, 1, ..., nlags. Shape (nlags+1,).

    Returns
    -------
    pacf (numpy.ndarray): The partial autocorrelations for lags 0, 1, ..., nlags. Shape (nlags+1,).
    """
    nlags = len(r)
    pacf = [1.0]
    for k in range(1, nlags):
        #print("Yule-Walker lag: {}".format(k))
        r_temp = r[: k + 1]
        R = scipy.linalg.toeplitz(r_temp[:-1])
        try:
            rho = np.linalg.solve(R, r_temp[1:])
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                warnings.warn("Matrix is singular. Using pinv.", ValueWarning)
                rho = np.linalg.pinv(R) @ r_temp[1:]
            else:
                raise
        pacf.append(rho[-1])
    pacf = np.array(pacf)

    return pacf



def calculate_cross_covariances(
    x,
    y,
    sd_x,
    sd_y,
    k,
    confint_coef=None,
    adjust_denominator=False,
):
    arr_corr_left = x[:-k]
    arr_corr_right = y[k:]
    n = len(arr_corr_left) + k
    r = (1 / (n - k * adjust_denominator)) * np.correlate(
        arr_corr_left, arr_corr_right
    )[0]
    ccf = r / (sd_x * sd_y)
    if confint_coef is None:
        return r, ccf
    confint = confint_coef * np.sqrt(1.0 / n)
    return r, ccf, confint




def calculate_crosscorrelations(
    x,
    y,
    nlags=20,
    alpha=0.05,
    x_equals_y=True,
    adjust_denominator=False,
    negative_lags=False,
):
    
    sd_x = x.std(ddof=0)
    sd_y = y.std(ddof=0)
    x -= x.mean()
    y -= y.mean()
    n = len(x)
    cross_covariance = (1 / n) * np.correlate(x, y)[0]
    cross_correlation = cross_covariance / (sd_x * sd_y)
    confint_coef = scipy.stats.norm.ppf(1.0 - (alpha / 2.0))

    ccf = np.zeros(nlags + 1, np.float64)
    ccf[0] = cross_correlation
    r = np.zeros(nlags + 1, np.float64)
    r[0] = cross_covariance
    confint = np.zeros(nlags + 1, np.float64)
    confint[0] = cross_correlation
    for k in range(1, nlags + 1):
        r_one_lag, ccf_one_lag, confint_one_lag = calculate_cross_covariances(
            x,
            y,
            sd_x,
            sd_y,
            k,
            confint_coef=confint_coef,
            adjust_denominator=adjust_denominator,
        )
        r[k] = r_one_lag
        ccf[k] = ccf_one_lag
        confint[k] = confint_one_lag
        #print("Construction lag: {}, nb datapoints={}".format(k, n))
    if (not negative_lags) or x_equals_y:
        pccf = pacf_yule_walker(r)

    if negative_lags and x_equals_y:
        ccf_neg = np.flip(ccf)[:-1]
        pccf_neg = np.flip(pccf)[:-1]
        confint_neg = np.flip(confint)[:-1]
        ccf = np.concatenate([ccf_neg, ccf])
        pccf = np.concatenate([pccf_neg, pccf])
        confint = np.concatenate([confint_neg, confint])
    elif negative_lags and (not x_equals_y):
        ccf_neg = np.zeros(nlags, np.float64)
        confint_neg = np.zeros(nlags, np.float64)
        for k in range(1, nlags + 1):
            ccf_one_lag, confint_one_lag = calculate_cross_covariances(
                y,
                x,
                sd_y,
                sd_x,
                k,
                confint_coef=confint_coef,
                adjust_denominator=adjust_denominator,
            )[1:]
            ccf_neg[k - 1] = ccf_one_lag
            confint_neg[k - 1] = confint_one_lag
            #print("Construction lag: {}, nb datapoints={}".format(-k, n))
        ccf_neg = np.flip(ccf_neg)
        confint_neg = np.flip(confint_neg)
        ccf = np.concatenate([ccf_neg, ccf])
        confint = np.concatenate([confint_neg, confint])
        pccf = cross_correlations_ols(
            x,
            y,
            cross_correlation,
            nlags,
            adjust_denominator=adjust_denominator,
            negative_lags=negative_lags,
        )
    
    return ccf, pccf, confint


def sum_of_exponentials(x, *params):
    result = 0
    for i in range(0, len(params), 2):
        result += params[i] * np.exp(-params[i+1] * x)
    return result


def sum_of_power_laws(x, *params):
    result = 0
    for i in range(0, len(params), 2):
        result += params[i] * ((1 / x) ** params[i+1])
    return result


def fit_exponential(x, y):
    # Initial guess for the parameters
    initial_guess = [1, 1]
    # Fit the data to the sum of exponentials function
    fit_params = curve_fit(sum_of_exponentials, x, y, p0=initial_guess, maxfev=1000000)[0]
    # Generate fitted curve using the obtained parameters
    y_fit = sum_of_exponentials(x, *fit_params)
    equation_str = ' + '.join([f"{fit_params[2 * i]:.4f} * \exp{{(-{fit_params[2 * i + 1]:.4f}x)}}" for i in range(int(len(fit_params) / 2))])
    equation_str = f"Exponential Fit: ${equation_str}$"
    equation_str = equation_str.replace('+ -', '-')
    return y_fit, equation_str


def fit_power(x, y):
    # Initial guess for the parameters
    initial_guess = [1, 0.5]
    # Fit the data to the sum of exponentials function
    fit_params = curve_fit(sum_of_power_laws, x, y, p0=initial_guess, maxfev=1000000)[0]
    # Generate fitted curve using the obtained parameters
    y_fit = sum_of_power_laws(x, *fit_params)
    equation_str = ' + '.join(["{:.4f} * (1/x)^{{{:.4f}}}".format(fit_params[2 * i], fit_params[2 * i + 1]) for i in range(int(len(fit_params) / 2))])
    equation_str = f"Power Fit: ${equation_str}$"
    equation_str = equation_str.replace('+ -', '-')
    return y_fit, equation_str


def plot_ccf_pccf(
    ccf,
    pccf,
    confint,
    alpha,
    x_equals_y=True,
    negative_lags=False,
    fit_type=None
):

    y_margin = 0.05
    max_nlags_stem = 50
    color = "royalblue"
    color_stemlines = color
    color_markerline = color
    str_confidence_level = str(np.round(100 * (1 - alpha), decimals=0)).split(".0")[0]
    ccf_ylabel = "ACF" if x_equals_y else "CCF"
    pccf_ylabel = "PACF" if x_equals_y else "PCCF"
    suptitle = "Autocorrelations" if x_equals_y else "Cross-Correlations"

    if negative_lags:
        nlags = len(ccf)
        x = np.array(range(1, int((nlags + 1) / 2)))
        x = np.concatenate([-np.flip(x), np.array([0]), x])
        ccf_temp = np.concatenate(
            [ccf[: int((nlags - 1) / 2)], ccf[int((nlags - 1) / 2) + 1 :]]
        )
        pccf_temp = np.concatenate(
            [pccf[: int((nlags - 1) / 2)], pccf[int((nlags - 1) / 2) + 1 :]]
        )
        confint_temp = np.concatenate(
            [confint[: int((nlags - 1) / 2)], confint[int((nlags - 1) / 2) + 1 :]]
        )
        max_abs_value = max(
            abs(min(min(ccf_temp), min(pccf_temp), min(confint_temp))),
            abs(max(max(ccf_temp), max(pccf_temp), max(confint_temp))),
        )
        max_value_lag_0 = max(
            abs(ccf[int((nlags - 1) / 2)]),
            abs(pccf[int((nlags - 1) / 2)]),
            abs(confint[int((nlags - 1) / 2)]),
        )
        if max_value_lag_0 <= 0.99:
            max_abs_value = max(max_abs_value, max_value_lag_0)
        confint[int((nlags - 1) / 2)] = (
            confint[int((nlags - 1) / 2) - 1] + confint[int((nlags - 1) / 2) + 1]
        ) / 2
        max_y_value = max_abs_value + y_margin
        min_y_value = -max_abs_value - y_margin
        fit_type = None # fit not implemented for negative lags
    else:
        ccf = ccf[1:]
        pccf = pccf[1:]
        confint = confint[1:]
        nlags = len(ccf)
        x = np.array(range(1, nlags + 1))
        max_abs_value = max(
            abs(min(min(ccf), min(pccf), min(confint))),
            abs(max(max(ccf), max(pccf), max(confint))),
        )
        max_y_value = max_abs_value + y_margin
        min_y_value = -max_abs_value - y_margin
    x_margin = max(0.5, (len(ccf) + 1) / 60)
    x_min = min(x) - x_margin
    x_max = max(x) + x_margin

    if fit_type is not None:
        if fit_type == 'exp':
            ccf_fit, ccf_label = fit_exponential(x, ccf)
            pccf_fit, pccf_label = fit_exponential(x, pccf)
        elif fit_type == 'power':
            ccf_fit, ccf_label = fit_power(x, ccf)
            pccf_fit, pccf_label = fit_power(x, pccf)
        else:
            raise ValueError("'{}' not implemented yet for fitting.")
        ccf_label = 'CCF ' + ccf_label
        pccf_label = 'PCCF ' + pccf_label

    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(8, 5), gridspec_kw={'hspace': 0})

    # Plot CCF using stem plot
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    if nlags <= max_nlags_stem:
        markerline, stemlines, _ = ax1.stem(x, ccf, linefmt="-", basefmt=" ")
        plt.setp(stemlines, "color", color_stemlines)  # Set stem line color
        plt.setp(markerline, "color", color_markerline)  # Set marker line color
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        if negative_lags:
            x_ticks = np.array([i for i in x if not i % 2])
        else:
            x_ticks = np.array([i for i in x if i % 2])
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticks)
    else:
        ax1.scatter(x, ccf, color=color, marker="o", s=10)
    if fit_type is not None:
        ax1.plot(x, ccf_fit, 'r--', color='seagreen', label=ccf_label)
    ax1.hlines(confint, x_min, x_max, color='crimson', linestyle='--', linewidth=0.6, label="{}% confidence interval".format(str_confidence_level))
    ax1.hlines(-confint, x_min, x_max, color='crimson', linestyle='--', linewidth=0.6)
    ax1.set_xlim(min(x) - x_margin, max(x) + x_margin)
    ax1.set_ylabel(ccf_ylabel)
    ax1.set_ylim(min_y_value, max_y_value)

    # Plot PCCF using stem plot
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    if nlags <= max_nlags_stem:
        markerline, stemlines, _ = ax2.stem(
            x, pccf, linefmt="-", basefmt=" "
        )
        plt.setp(stemlines, "color", color_stemlines)  # Set stem line color
        plt.setp(markerline, "color", color_markerline)  # Set marker line color
    else:
        ax2.scatter(x, pccf, color=color, marker="o", s=10)
    if fit_type is not None:
        ax2.plot(x, pccf_fit, 'r--', color='seagreen', label=pccf_label)
    ax2.hlines(confint, x_min, x_max, color='crimson', linestyle='--', linewidth=0.6)
    ax2.hlines(-confint, x_min, x_max, color='crimson', linestyle='--', linewidth=0.6)
    ax2.set_xlabel("Lag")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylabel(pccf_ylabel)
    ax2.set_ylim(min_y_value, max_y_value)

    if fit_type is not None:
        # Get handles and labels from both legends
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = [handles1[1]]
        labels = [labels1[1]]
        handles_ccf = [handles1[0]]
        labels_ccf = [labels1[0]]
        handles_pccf = [handles2[0]]
        labels_pccf = [labels2[0]]
        ax1.legend(handles_ccf, labels_ccf, loc='lower right')
        ax2.legend(handles_pccf, labels_pccf, loc='lower right')
    else:
        handles, labels = ax1.get_legend_handles_labels()

    fig.legend(handles, labels, bbox_to_anchor=(0.65, 0.9))
    fig.suptitle(
        "{}".format(suptitle),
        fontsize=15,
        y=0.943,
    )
    fig.subplots_adjust(hspace=0.2, top=0.81)

    return fig


def plot_crosscorrelations(
    x,
    y,
    nlags=20,
    alpha=0.05,
    adjust_denominator=False,
    negative_lags=False,
    fit_type=None
):
    
    x_equals_y = np.array_equal(x, y)
    ccf, pccf, confint = calculate_crosscorrelations(
        x,
        y,
        nlags=nlags,
        alpha=alpha,
        x_equals_y=x_equals_y,
        adjust_denominator=adjust_denominator,
        negative_lags=negative_lags,
    )

    fig = plot_ccf_pccf(
        ccf,
        pccf,
        confint,
        alpha=alpha,
        x_equals_y=x_equals_y,
        negative_lags=negative_lags,
        fit_type=fit_type,
    )

    return fig


def extract_data_from_axes(ax):
    # Extracting data from the resulting figure
    scatter_plots = [artist for artist in ax.get_children()]

    for _, scatter_plot in enumerate(scatter_plots, start=1):
        try:
            offsets = scatter_plot.get_offsets()
            y_data = offsets[:, 1]  # Get y coordinates

        except AttributeError:
            pass
    return y_data

