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
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    if len(y.shape) == 1:
        y = y.reshape((y.shape[0], 1))
    n_sim = x.shape[1]

    arr_corr_left = x[:-k, :]
    arr_corr_right = y[k:, :]
    n = arr_corr_left.shape[0] + k

    r = np.zeros(n_sim)
    for i in range(n_sim):
        r[i] = (1 / (n - k * adjust_denominator)) * np.correlate(arr_corr_left[:, i], arr_corr_right[:, i])[0]
    ccf = r / (sd_x * sd_y)
    if confint_coef is None:
        if n_sim == 1:
            return r[0], ccf[0]
        return r, ccf
    confint = np.array([confint_coef * np.sqrt(1.0 / n)] * n_sim)
    if n_sim == 1:
        return r[0], ccf[0], confint[0]
    return r, ccf, confint


def calculate_crosscorrelations(
    x,
    y,
    nlags=20,
    alpha=0.05,
    x_equals_y=True,
    adjust_denominator=False,
    negative_lags=False,
    compute_pccf=True,
):
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    if len(y.shape) == 1:
        y = y.reshape((y.shape[0], 1))
    n_sim = x.shape[1]

    sd_x = x.std(ddof=0, axis=0)
    sd_y = y.std(ddof=0, axis=0)
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    n = x.shape[0]
    cross_covariances = []
    for i in range(n_sim):
        cross_covariances.append((1 / n) * np.correlate(x[:, i], y[:, i])[0])
    cross_correlations = cross_covariances / (sd_x * sd_y)
    confint_coef = scipy.stats.norm.ppf(1.0 - (alpha / 2.0))
    ccf = np.zeros((nlags + 1, n_sim), np.float64)
    ccf[0, :] = cross_correlations
    r = np.zeros((nlags + 1, n_sim), np.float64)
    r[0, :] = cross_covariances
    confint = np.zeros((nlags + 1, n_sim), np.float64)
    confint[0, :] = cross_correlations

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
        r[k, :] = r_one_lag
        ccf[k, :] = ccf_one_lag
        confint[k, :] = confint_one_lag
    if compute_pccf and ((not negative_lags) or x_equals_y):
        pccf = np.zeros((nlags + 1, n_sim), np.float64)
        for i in range(n_sim):
            pccf[:, i] = pacf_yule_walker(r[:, i])

    if negative_lags and x_equals_y:
        ccf_neg = np.flip(ccf, axis=0)[:-1]
        ccf = np.concatenate([ccf_neg, ccf], axis=0)
        confint_neg = np.flip(confint, axis=0)[:-1]
        confint = np.concatenate([confint_neg, confint], axis=0)
        if compute_pccf:
            pccf_neg = np.flip(pccf, axis=0)[:-1]
            pccf = np.concatenate([pccf_neg, pccf], axis=0)
    elif negative_lags and (not x_equals_y):
        ccf_neg = np.zeros((nlags, n_sim), np.float64)
        confint_neg = np.zeros((nlags, n_sim), np.float64)
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
            ccf_neg[k - 1, :] = ccf_one_lag
            confint_neg[k - 1, :] = confint_one_lag
        ccf_neg = np.flip(ccf_neg, axis=0)
        confint_neg = np.flip(confint_neg, axis=0)
        ccf = np.concatenate([ccf_neg, ccf], axis=0)
        confint = np.concatenate([confint_neg, confint], axis=0)
        if compute_pccf:
            pccf = np.zeros((2 * nlags + 1, n_sim), np.float64)
            for i in range(n_sim):
                pccf[:, i] = cross_correlations_ols(
                    x[:, i],
                    y[:, i],
                    cross_correlations[i],
                    nlags,
                    adjust_denominator=adjust_denominator,
                    negative_lags=negative_lags,
                )

    if n_sim == 1:
        ccf = np.squeeze(ccf, axis=1)
        confint = np.squeeze(confint, axis=1)
    if not compute_pccf:
        return ccf, confint
    if n_sim == 1:
        pccf = np.squeeze(pccf, axis=1)
    return ccf, pccf, confint


def sum_of_exponentials(x, *params):
    result = 0
    for i in range(0, len(params), 2):
        result += params[i] * np.exp(-params[i+1] * x)
    return result


def sum_of_power_laws_2_params(x, *params):
    result = 0
    for i in range(0, len(params), 2):
        result += params[i] * ((1 / x) ** params[i+1])
    return result


def sum_of_power_laws_3_params(x, *params):
    result = 0
    for i in range(0, len(params), 3):
        result += params[i] * ((1 / (x + params[i+2])) ** params[i+1])
    return result


def fit_exponential(x, y):
    # Initial guess for the parameters
    initial_guess = [1, 1]
    # Fit the data to the sum of exponentials function
    fit_params = curve_fit(sum_of_exponentials, x, y, p0=initial_guess, maxfev=1000000)[0]
    # Generate fitted curve using the obtained parameters
    y_fit = sum_of_exponentials(x, *fit_params)
    rss = np.round(np.sum((y - y_fit) ** 2), decimals=3)
    equation_str = ' + '.join([f"{fit_params[2 * i]:.4f} * \exp{{(-{fit_params[2 * i + 1]:.4f}x)}}" for i in range(int(len(fit_params) / 2))])
    equation_str = f"Exponential Fit: ${equation_str}$, $RSS={rss}$"
    equation_str = equation_str.replace('+ -', '-')
    return y_fit, equation_str


def fit_power(x, y, nb_params=2):
    # Initial guess for the parameters
    initial_guess = [1, 0.5] if nb_params == 2 else [1, 0.5, 10]
    # Fit the data to the sum of exponentials function
    if nb_params == 2:
        fit_params = curve_fit(sum_of_power_laws_2_params, x, y, p0=initial_guess, maxfev=1000000)[0]
        y_fit = sum_of_power_laws_2_params(x, *fit_params)
        equation_str = ' + '.join(["{:.4f} * (1/x)^{{{:.4f}}}".format(fit_params[2 * i], fit_params[2 * i + 1]) for i in range(int(len(fit_params) / 2))])
    else:
        fit_params = curve_fit(sum_of_power_laws_3_params, x, y, p0=initial_guess, maxfev=1000000)[0]
        y_fit = sum_of_power_laws_3_params(x, *fit_params)
        equation_str = ' + '.join(["{:.4f} * (1/(x + {{{:.4f}}}))^{{{:.4f}}}".format(fit_params[3 * i], fit_params[3 * i + 1], fit_params[3 * i + 2]) for i in range(int(len(fit_params) / 3))])
    rss = np.round(np.sum((y - y_fit) ** 2), decimals=3)
    equation_str = f"Power Fit: ${equation_str}$, $RSS={rss}$"
    equation_str = equation_str.replace('+ -', '-')
    return y_fit, equation_str


def plot_ccf_pccf(
    ccf,
    pccf,
    confint,
    ccf_std,
    pccf_std,
    alpha,
    x_equals_y=True,
    negative_lags=False,
    fit_type=None,
    nb_params=2,
    n_sim=None
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
        confint_temp = np.concatenate(
            [confint[: int((nlags - 1) / 2)], confint[int((nlags - 1) / 2) + 1 :]]
        )
        if pccf is not None:
            pccf_temp = np.concatenate(
                [pccf[: int((nlags - 1) / 2)], pccf[int((nlags - 1) / 2) + 1 :]]
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
        else:
            max_abs_value = max(
                abs(min(min(ccf_temp), min(confint_temp))),
                abs(max(max(ccf_temp), max(confint_temp))),
            )
            max_value_lag_0 = max(
                abs(ccf[int((nlags - 1) / 2)]),
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
        if pccf is not None:
            pccf = pccf[1:]
        confint = confint[1:]
        nlags = len(ccf)
        x = np.array(range(1, nlags + 1))
        if pccf is not None:
            max_abs_value = max(
                abs(min(min(ccf), min(pccf), min(confint))),
                abs(max(max(ccf), max(pccf), max(confint))),
            )
        else:
            max_abs_value = max(
                abs(min(min(ccf), min(confint))),
                abs(max(max(ccf), max(confint))),
            )
        max_y_value = max_abs_value + y_margin
        min_y_value = -max_abs_value - y_margin
    x_margin = max(0.5, (len(ccf) + 1) / 60)
    x_min = min(x) - x_margin
    x_max = max(x) + x_margin

    if fit_type is not None:
        if fit_type == 'exp':
            ccf_fit, ccf_label = fit_exponential(x, ccf)
            ccf_label = 'CCF ' + ccf_label
            if pccf is not None:
                pccf_fit, pccf_label = fit_exponential(x, pccf)
                pccf_label = 'PCCF ' + pccf_label
        elif fit_type == 'power':
            ccf_fit, ccf_label = fit_power(x, ccf, nb_params=nb_params)
            ccf_label = 'CCF ' + ccf_label
            if pccf is not None:
                pccf_fit, pccf_label = fit_power(x, pccf, nb_params=nb_params)
                pccf_label = 'PCCF ' + pccf_label
        else:
            raise ValueError("'{}' not implemented yet for fitting.")

    if pccf is not None:
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(8, 5), gridspec_kw={'hspace': 0})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 3))

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
    if ccf_std is not None:
        ccf_std = ccf_std[1:]
        ax1.fill_between(x, ccf - ccf_std, ccf + ccf_std, alpha=0.2, label='Standard error ({} simulations)'.format(n_sim))
    if fit_type is not None:
        ax1.plot(x, ccf_fit, 'r--', color='seagreen', label=ccf_label)
    ax1.hlines(confint, x_min, x_max, color='crimson', linestyle='--', linewidth=0.6, label="{}% confidence interval".format(str_confidence_level))
    ax1.hlines(-confint, x_min, x_max, color='crimson', linestyle='--', linewidth=0.6)
    ax1.set_xlim(min(x) - x_margin, max(x) + x_margin)
    ax1.set_ylabel(ccf_ylabel)
    ax1.set_ylim(min_y_value, max_y_value)

    if pccf is not None:
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

    if pccf_std is not None:
        pccf_std = pccf_std[1:]
        ax2.fill_between(x, pccf - pccf_std, pccf + pccf_std, alpha=0.2)

    if (ccf_std is None) and (pccf is not None):
        bbox_to_anchor = (0.51, 0.9)
        ncol = 1
    elif (ccf_std is not None) and (pccf is not None):
        bbox_to_anchor = (0.51, 0.9)
        ncol = 2
    elif (ccf_std is None) and (pccf is None):
        bbox_to_anchor = (0.51, 0.87)
        ncol = 1
    elif (ccf_std is not None) and (pccf is None):
        bbox_to_anchor = (0.51, 0.87)
        ncol = 2

    if fit_type is not None:
        # Get handles and labels from both legends
        handles1, labels1 = ax1.get_legend_handles_labels()
        if ccf_std is not None:
            handles = [handles1[0], handles1[2]]
            labels = [labels1[0], labels1[2]]
            handles_ccf = [handles1[1]]
            labels_ccf = [labels1[1]]
        else:
            handles = [handles1[1]]
            labels = [labels1[1]]
            handles_ccf = [handles1[0]]
            labels_ccf = [labels1[0]]
        if pccf is not None:
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles_pccf = [handles2[0]]
            labels_pccf = [labels2[0]]
            ax2.legend(handles_pccf, labels_pccf, loc='lower right')
        ax1.legend(handles_ccf, labels_ccf, loc='lower right')
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=bbox_to_anchor, ncol=ncol)
    else:
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=bbox_to_anchor, ncol=ncol)

    if pccf is not None:
        top = 0.81
    else:
        top = 0.72
    fig.suptitle(
        "{}".format(suptitle),
        fontsize=15,
        y=0.943,
    )
    fig.subplots_adjust(hspace=0.2, top=top)

    return fig


def plot_crosscorrelations(
    x,
    y,
    nlags=20,
    alpha=0.05,
    adjust_denominator=False,
    negative_lags=False,
    fit_type=None,
    nb_params=2,
):
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    if len(y.shape) == 1:
        y = y.reshape((y.shape[0], 1))
    n_sim = x.shape[1]

    x_equals_y = np.array_equal(x, y)
    compute_pccf = nlags <= 100
    results = calculate_crosscorrelations(
        x,
        y,
        nlags=nlags,
        alpha=alpha,
        x_equals_y=x_equals_y,
        adjust_denominator=adjust_denominator,
        negative_lags=negative_lags,
        compute_pccf=compute_pccf,
    )

    if compute_pccf:
        ccf, pccf, confint = results
    else:
        ccf, confint = results
        pccf = None
    if n_sim == 1:
        ccf = ccf.reshape((ccf.shape[0], 1))
        confint = confint.reshape((confint.shape[0], 1))
        if compute_pccf:
            pccf = pccf.reshape((pccf.shape[0], 1))

    if n_sim > 1:
        ccf_std = ccf.std(ddof=0, axis=1)
        if compute_pccf:
            pccf_std = pccf.std(ddof=0, axis=1)
            pccf = pccf.mean(axis=1)
        else:
            pccf_std = None
    else:
        ccf_std = None
        pccf_std = None
        if compute_pccf:
            pccf = pccf.mean(axis=1)

    ccf = ccf.mean(axis=1)
    confint = confint[:, 0]

    fig = plot_ccf_pccf(
        ccf,
        pccf,
        confint,
        ccf_std,
        pccf_std,
        alpha=alpha,
        x_equals_y=x_equals_y,
        negative_lags=negative_lags,
        fit_type=fit_type,
        nb_params=nb_params,
        n_sim=n_sim,
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

