import datetime
import string
import warnings

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.ticker import MaxNLocator
from statsmodels.compat.numpy import lstsq
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.tsatools import lagmat


class DataFrameTools:
    @staticmethod
    def construct_datetime_columns(
        df, construct_time=True, construct_date=True, construct_hour=True
    ):
        if construct_time:
            df["Time Datetime"] = df["Time"].apply(
                lambda value: datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            )
        if construct_date:
            df["Date"] = df["Time"].apply(lambda value: value.split(" ")[0])
        if construct_hour:
            df["Hour Datetime"] = df["Hour"].apply(
                lambda value: datetime.datetime.strptime(value, "%H:%M:%S")
            )
        return df


class CrossCorrelationTools:
    def __init__(self, df):
        self.df = df.copy()

    def calculate_cross_correlation_ols(
        self,
        x,
        y,
        xlags,
        ylags,
        x0,
        y0,
        k,
        adjust_denominator=False,
        adjust_daily=False,
    ):
        xparams = lstsq(xlags[k - 1 :, :k], x0[k - 1 :], rcond=None)[0]
        xresiduals = x0[k - 1 :] - xlags[k - 1 :, :k] @ xparams
        xresiduals = pd.Series(
            data=xresiduals.reshape(xresiduals.shape[0]),
            index=x.iloc[k - 1 :].index,
            name=x.name,
        )

        yparams = lstsq(ylags[k - 1 :, :k], y0[k - 1 :], rcond=None)[0]
        yresiduals = y0[k - 1 :] - ylags[k - 1 :, :k] @ yparams
        yresiduals = pd.Series(
            data=yresiduals.reshape(yresiduals.shape[0]),
            index=y.iloc[k - 1 :].index,
            name=y.name,
        )

        sd_xresiduals = xresiduals.std(ddof=0)
        sd_yresiduals = yresiduals.std(ddof=0)
        xresiduals -= xresiduals.mean()
        yresiduals -= yresiduals.mean()

        pccf = self.calculate_cross_covariances(
            xresiduals,
            yresiduals,
            sd_xresiduals,
            sd_yresiduals,
            k,
            confint_coef=None,
            adjust_denominator=adjust_denominator,
            adjust_daily=adjust_daily,
        )[1]

        return pccf

    def cross_correlations_ols(
        self,
        x,
        y,
        cross_correlation,
        nlags,
        adjust_denominator=False,
        adjust_daily=False,
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
            print("PCCF lag: {}".format(k))
            ccf = self.calculate_cross_correlation_ols(
                x,
                y,
                xlags,
                ylags,
                x0,
                y0,
                k,
                adjust_denominator=adjust_denominator,
                adjust_daily=adjust_daily,
            )
            pccf[k] = ccf

        if negative_lags:
            pccf_neg = np.zeros(nlags, np.float64)
            for k in range(1, nlags + 1):
                print("PCCF lag: {}".format(-k))
                ccf = self.calculate_cross_correlation_ols(
                    y,
                    x,
                    ylags,
                    xlags,
                    y0,
                    x0,
                    k,
                    adjust_denominator=adjust_denominator,
                    adjust_daily=adjust_daily,
                )
                pccf_neg[k - 1] = ccf
            pccf_neg = np.flip(pccf_neg)
            pccf = np.concatenate([pccf_neg, pccf])

        return pccf

    @staticmethod
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
            print("Yule-Walker lag: {}".format(k))
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

    @staticmethod
    def calculate_cross_covariances(
        x,
        y,
        sd_x,
        sd_y,
        k,
        confint_coef=None,
        adjust_denominator=False,
        adjust_daily=False,
    ):
        df = x.iloc[:-k].reset_index().rename(columns={x.name: "x-values"})
        df_lagged = (
            y.iloc[k:]
            .reset_index()
            .rename(columns={"Time": "Time Lagged", y.name: "y-values"})
        )
        df = pd.concat([df, df_lagged], axis=1)
        if adjust_daily:
            df = df.loc[
                df.apply(
                    lambda row: row["Time"].split(" ")[0]
                    == row["Time Lagged"].split(" ")[0],
                    axis=1,
                )
            ].copy()
        arr_corr_left = np.array(df["x-values"].to_list())
        arr_corr_right = np.array(df["y-values"].to_list())
        n = len(arr_corr_left) + k
        r = (1 / (n - k * adjust_denominator)) * np.correlate(
            arr_corr_left, arr_corr_right
        )[0]
        ccf = r / (sd_x * sd_y)
        if confint_coef is None:
            return r, ccf
        confint = confint_coef * np.sqrt(1.0 / n)
        return r, ccf, confint

    def calculate_autocorrelations(
        self,
        x,
        y,
        nlags=20,
        alpha=0.05,
        x_equals_y=True,
        adjust_denominator=False,
        adjust_daily=False,
        negative_lags=False,
    ):
        """
        Compute the autocorrelations and partial cross-correlations between x and y (or autocorrelations if x and y are identical).

        Parameters
        ----------
        x (pandas.core.series.Series): A pandas Series. The index should be named 'Time' and contain the
            times associated with the x-values as strings with format '%Y-%m-%d %H:%M:%S'.
        y (pandas.core.series.Series): A pandas Series. The index should be named 'Time' and contain the
            times associated with the y-values as strings with format '%Y-%m-%d %H:%M:%S'.
        nlags (int, optional): The number of lags to return cross-correlations for. Default is 20.
        alpha (float, optional): The confidence level for the confidence intervals of the cross-correlations.
            For instance if alpha=.05, 95% confidence intervals are returned where the standard deviation
            is computed according to 1/sqrt(number of observations). Default is 0.05.
        x_equals_y (bool, optional): If True, the time series x and y are considered to be equal, to that we
            compute the autocorrelations by solving the Yule-Walker equations. If False, they are considered
            not to be equal, and we compute the cross-correlations using OLS regressions. Default is True.
            Caution! If x and y are not equal but x_equals_y=True, then the output partial cross-correlations
            will be inconsistent and unreliable.
        adjust_denominator (bool, optional): Determines denominator in estimate of cross-correlation function
            at lag k. If False, the denominator is n=len(returns), if True the denominator is n-k. Default is False.
        adjust_daily (bool, optional): If True, the cross-covariances used in estimating the
            cross-correlations are computed by multiplying returns on same days only. This implies that
            the larger the lag, the less data we use to compute the cross-correlations. Default is False.

        Returns
        -------
        ccf (numpy.ndarray): The cross-correlations for lags 0, 1, ..., nlags computed using Pearson correlation
            coefficients. The cross-correlations are calculated by lagging the time series y: That is, when
            the lag is 1, the cross-correlation is an estimate of Cov(X_t, Y_{t+1}) / (SD(X)SD(Y)). Shape (nlags+1,).
        pccf (numpy.ndarray): The partial cross-correlations for lags 0, 1, ..., nlags. Shape (nlags+1,).
            The partial-cross correlations are calculated:
                - By solving the Yule Walker equations if x and y are identical,
                - Using OLS regressions otherwise.
        confint (numpy.ndarray): The upper bounds of the symmetric confidence intervals for the
            cross-correlations of lags 0, 1, ..., nlags. That is, the confidence interval of
            cross-correlation of lag k is given by [-confint[k], confint[k]]. Shape (nlags+1,).
        """
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
            r_one_lag, ccf_one_lag, confint_one_lag = self.calculate_cross_covariances(
                x,
                y,
                sd_x,
                sd_y,
                k,
                confint_coef=confint_coef,
                adjust_denominator=adjust_denominator,
                adjust_daily=adjust_daily,
            )
            r[k] = r_one_lag
            ccf[k] = ccf_one_lag
            confint[k] = confint_one_lag
            print("Construction lag: {}, nb datapoints={}".format(k, n))
        if (not negative_lags) or x_equals_y:
            pccf = self.pacf_yule_walker(r)

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
                ccf_one_lag, confint_one_lag = self.calculate_cross_covariances(
                    y,
                    x,
                    sd_y,
                    sd_x,
                    k,
                    confint_coef=confint_coef,
                    adjust_denominator=adjust_denominator,
                    adjust_daily=adjust_daily,
                )[1:]
                ccf_neg[k - 1] = ccf_one_lag
                confint_neg[k - 1] = confint_one_lag
                print("Construction lag: {}, nb datapoints={}".format(-k, n))
            ccf_neg = np.flip(ccf_neg)
            confint_neg = np.flip(confint_neg)
            ccf = np.concatenate([ccf_neg, ccf])
            confint = np.concatenate([confint_neg, confint])
            pccf = self.cross_correlations_ols(
                x,
                y,
                cross_correlation,
                nlags,
                adjust_denominator=adjust_denominator,
                adjust_daily=adjust_daily,
                negative_lags=negative_lags,
            )

        print("__________________")
        # pccf = self.cross_correlations_ols(x, y, cross_correlation, nlags, adjust_denominator=adjust_denominator, adjust_daily=adjust_daily, negative_lags=negative_lags)
        # pd.Series(pccf).to_excel('E:\\OneDrive\\Documents\\MIASHS 2022-10\\ENSAE_2023-2024\\Calibration VIX-SPX\\Rapport\\Validation_ols\\Test_PCCF_OLS2.xlsx')

        return ccf, pccf, confint

    def plot_ccf_pccf(
        self,
        ccf,
        pccf,
        confint,
        start_date,
        end_date,
        alpha,
        x_equals_y=True,
        negative_lags=False,
        upload_path=None,
    ):
        """
        Plot the CCF and PCCF of the time series.

        Parameters
        ----------
        ccf (numpy.ndarray): The cross-correlations for lags 0, 1, ..., len(ccf)-1.
        pccf (numpy.ndarray): The partial cross-correlations for lags 0, 1, ..., len(pccf)-1.
        confint (numpy.ndarray): The upper bounds of the symmetric confidence intervals for the
            cross-correlations of lags 0, 1, ..., len(confint)-1. That is, the confidence interval of
            cross-correlation of lag k should be given by [-confint[k], confint[k]].
        start_date (datetime.datetime): The start date to be considered when calculating the cross-correlations.
        end_date (datetime.datetime): The end date to be considered when calculating the cross-correlations.
        alpha (float): The confidence level for the confidence intervals of the cross-correlations.
        x_equals_y (bool, optional): If True, the correlations passed as inputs are considered to be autocorrelations
            of one time series instead of cross-correlations of two distinct time series. Otherwise, the correlations
            are considered to be cross-correlations of two distinct time series. This parameter is only used to adjust
            the y-labels and the suptitle of the figure. Default is True.
        upload_path (string or None, optional): The path where to save the figure. If not None, the figure
            is saved according to the input path. If None, the figure is not saved. Default is None.

        Returns
        -------
        None.
        """
        y_margin = 0.005
        max_nlags_stem = 50
        color = "#1f77b4"
        color_stemlines = color
        color_markerline = color
        str_confidence_level = str(np.round(100 * (1 - alpha), decimals=0)).split(".0")[
            0
        ]
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

        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(7, 4))

        # Plot CCF using stem plot
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        if nlags <= max_nlags_stem:
            markerline, stemlines, baseline = ax1.stem(x, ccf, linefmt="-", basefmt=" ")
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
        ax1.set_xlim(min(x) - x_margin, max(x) + x_margin)
        ax1.set_ylabel(ccf_ylabel)
        ax1.set_ylim(min_y_value, max_y_value)
        ax1.plot(
            x,
            confint,
            color="red",
            linestyle="--",
            label="{}% confidence interval".format(str_confidence_level),
        )
        ax1.plot(x, -confint, color="red", linestyle="--")

        # Plot PCCF using stem plot
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        if nlags <= max_nlags_stem:
            markerline, stemlines, baseline = ax2.stem(
                x, pccf, linefmt="-", basefmt=" "
            )
            plt.setp(stemlines, "color", color_stemlines)  # Set stem line color
            plt.setp(markerline, "color", color_markerline)  # Set marker line color
        else:
            ax2.scatter(x, pccf, color=color, marker="o", s=10)
        ax2.set_xlabel("Lag (in minutes)")
        ax2.set_xlim(min(x) - x_margin, max(x) + x_margin)
        ax2.set_ylabel(pccf_ylabel)
        ax2.set_ylim(min_y_value, max_y_value)
        ax2.plot(x, confint, color="red", linestyle="--")
        ax2.plot(x, -confint, color="red", linestyle="--")

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.68, 0.874))
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        fig.suptitle(
            "{} from {} to {}".format(suptitle, start_date, end_date),
            fontsize=15,
            y=0.943,
        )
        fig.subplots_adjust(hspace=0.2, top=0.75)

        if upload_path is not None:
            plt.savefig(upload_path)
        plt.show()

    def plot_autocorrelations(
        self,
        start_date,
        end_date,
        filter_minutes=15,
        nlags=20,
        alpha=0.05,
        adjust_denominator=False,
        adjust_daily=False,
        transformation=lambda x: x,
        negative_lags=False,
        upload_path=None,
    ):
        """
        Construct the series of returns and plot the autocorrelations.

        Parameters
        ----------
        start_date (datetime.datetime): The start date to be considered when calculating the cross-correlations.
        end_date (datetime.datetime): The end date to be considered when calculating the cross-correlations.
        filter_minutes (int, optional): The number of minutes to be ignored at the beginning and end of each day. Default is 15
        nlags (int, optional): The number of lags to plot cross-correlation for. Default is 20.
        alpha (float, optional): The confidence level for the confidence intervals of the cross-correlations.
            For instance if alpha=.05, 95% confidence intervals are plotted where the standard deviation
            is computed according to 1/sqrt(number of observations). Default is 0.05.
        adjust_denominator (bool, optional): Determines denominator in estimate of cross-correlation function (CCF)
            at lag k. If False, the denominator is n=len(returns), if True the denominator is n-k. Default is False.
        adjust_daily (bool, optional): If True, the cross-covariances used in estimating the
            cross-correlations are computed by multiplying returns on same days only. This implies that
            the larger the lag, the less data we use to compute the cross-correlations. Default is False.
        transformation (func, optional): A function to be applied to the returns, e.g. the absolute value function.
            Default is the identity function.
        upload_path (string or None, optional): The path where to save the figure. If not None, the figure
            is saved according to the input path. If None, the figure is not saved. Default is None.

        Returns
        -------
        None.
        """
        df = self.df.copy()
        start_date = datetime.datetime(
            year=start_date.year,
            month=start_date.month,
            day=start_date.day,
            hour=0,
            minute=0,
        )
        end_date = datetime.datetime(
            year=end_date.year,
            month=end_date.month,
            day=end_date.day,
            hour=23,
            minute=59,
        )
        df = df.loc[
            (df["Time Datetime"] >= start_date) & (df["Time Datetime"] <= end_date)
        ].copy()
        df.reset_index(drop=True, inplace=True)
        start_date = df["Time Datetime"].iloc[0]
        end_date = df["Time Datetime"].iloc[-1]
        start_date = datetime.datetime(
            year=start_date.year,
            month=start_date.month,
            day=start_date.day,
            hour=0,
            minute=0,
        )
        end_date = datetime.datetime(
            year=end_date.year,
            month=end_date.month,
            day=end_date.day,
            hour=23,
            minute=59,
        )

        # Filter out the returns at the beginning and end of each day
        daily_start_date = datetime.datetime(
            year=1900, month=1, day=1, hour=9, minute=30
        )
        daily_end_date = datetime.datetime(year=1900, month=1, day=1, hour=16, minute=0)
        daily_start_date = daily_start_date + dateutil.relativedelta.relativedelta(
            minutes=filter_minutes
        )
        daily_end_date = daily_end_date + dateutil.relativedelta.relativedelta(
            minutes=-filter_minutes
        )
        df = df.loc[
            (df["Hour Datetime"] >= daily_start_date)
            & (df["Hour Datetime"] <= daily_end_date)
        ].copy()

        df.set_index("Time", inplace=True)
        x = df["Log-returns 1-min Period"]
        x.name = "x-values"
        # x = np.abs(x)
        x = transformation(x)

        y = df["Log-returns 1-min Period"]
        y.name = "y-values"
        y = y**2
        # y = transformation(y)

        x_equals_y = x.equals(y)
        ccf, pccf, confint = self.calculate_autocorrelations(
            x,
            y,
            nlags=nlags,
            alpha=alpha,
            x_equals_y=x_equals_y,
            adjust_denominator=adjust_denominator,
            adjust_daily=adjust_daily,
            negative_lags=negative_lags,
        )

        self.plot_ccf_pccf(
            ccf,
            pccf,
            confint,
            start_date,
            end_date,
            alpha=alpha,
            x_equals_y=x_equals_y,
            negative_lags=negative_lags,
            upload_path=upload_path,
        )

        # Voir s'il ne faut pas ajuster pacf_yuler
        # Changer les autocorrelations en cross-corrélations dans les noms de variables et commentaires
        # Booléen pour savoir si on veut des lags négatifs ou non
        # Vérifier qu'on est toujours cohérent avec statsmodels
        # Voir si on fait plusieurs classes pour chacun des faits stylisés, qui appellent la classe CrossCorrel
        # ou si on fait plusieurs fonctions dans la classe CrossCorrel, une pour chacune des graphiques
        # (CCF des rnedements, des rendements en valeur absolue...)
        # Voir comment traiter la transformation: left and right transformation, ??
        # Mettre à jour les docstrings avec les arguments supplémentaires: negative_lags notamment
        # Problème: cross-correl OLS très loin de Yule avec absolute returns
        # Changer les noms de variables qui sont des pacf


class MomentTools:
    def __init__(self, df):
        self.df = df.copy()

    @staticmethod
    def compute_log_returns(df, period, filter_minutes=15, normalize=False):
        """
        Adjust the log-returns contained in the dataframe df, so as to exclude the first log-return of each day. Indeed, we
        look at time series limited to trading days from 9:30am to 4:00pm, so that the first log-return is computed without
        taking into account 'after-hours' trading activity. See https://arxiv.org/abs/2311.07738 for a similar methodology.

        Parameters:
        - df (pandas.core.frame.DataFrame): Dataframe containing the data of interest. Should contain the following columns:
            'Date': The day of each bar as a string
            'Hour Datetime': The hour of each bar as a datetime.datetime. Should be with format
                             '%Y-%m-%d %H:%M:%S' with %Y-%m-%d=1900-01-01 (e.g., 1900-01-01 10:00:00)
            'Log-returns': The log-returns
        - period (int): The period of the log-returns. For instance, if period is equal to 2, the log-return on 2023-11-24
          at 10:00 a.m. is considered to have been calculated using the last price on 2023-11-24 at 10:00 a.m. and the
          last price two minutes prior, i.e. the last price on 2023-11-24 at 9:58 a.m.
        - filter_minutes (int, optional): The number of minutes to be ignored at the beginning and end of each day. Default is 15.
        - normalize (bool, optional): A boolean value indicating whether the log-returns should be normalized on a daily
          basis or not. Default is False.

        Returns:
        - An numpy.ndarray containing treated log-returns.
        """
        # Removing the first log-return of each day (since it was calculated using the price of the previous day)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Index"}, inplace=True)
        list_indices = df.groupby("Date").first()["Index"].to_list()
        df.drop(index=list_indices, inplace=True)
        df.drop("Index", axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)

        if filter_minutes != 0:
            # Filter out the returns at the beginning and end of each day
            daily_start_date = datetime.datetime(
                year=1900, month=1, day=1, hour=9, minute=30
            )
            daily_end_date = datetime.datetime(
                year=1900, month=1, day=1, hour=16, minute=0
            )
            daily_start_date = daily_start_date + dateutil.relativedelta.relativedelta(
                minutes=filter_minutes + period
            )
            daily_end_date = daily_end_date + dateutil.relativedelta.relativedelta(
                minutes=-filter_minutes
            )
            df = df.loc[
                (df["Hour Datetime"] >= daily_start_date)
                & (df["Hour Datetime"] <= daily_end_date)
            ].copy()

        if normalize:
            dict_daily_averages = df.groupby("Date").mean().to_dict()["Log-returns"]
            dict_daily_sd = df.groupby("Date").std().to_dict()["Log-returns"]
            return np.array(
                df.apply(
                    lambda row: (row["Log-returns"] - dict_daily_averages[row["Date"]])
                    / dict_daily_sd[row["Date"]],
                    axis=1,
                ).to_list()
            )
        else:
            return np.array(df["Log-returns"].to_list())

    @staticmethod
    def calculate_excess_moment(returns, moment=4):
        """
        Calculate the excess moment of order moment of the returns, where 'excess' means that the theoretical
        moment of order moment of the standard normal distribution is substracted from the empirical moment.

        Parameters:
        - returns (numpy.ndarray): Array containing the returns.
        - moment (int, optional): The order of the excess moment to be calculated. Default is 4,
          which means that by default the function returns the excess kurtosis of the returns.

        Returns:
        - excess_moment (float): The excess moment of order moment of the returns.
        """
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)

        # Calculate empirical moment
        numerator = np.mean((returns - mean_returns) ** moment)
        denominator = std_returns**moment
        empirical_moment = numerator / denominator

        # Calculate the theoretical moment of the standard normal distribution
        if moment % 2:
            moment_snd = 0
        else:
            moment_snd = np.math.factorial(moment) / (
                (2 ** (moment / 2)) * np.math.factorial(int(moment / 2))
            )

        excess_moment = empirical_moment - moment_snd
        return excess_moment

    def compute_moment_between_two_dates(
        self, df, start_date, end_date, moment=4, filter_minutes=15, normalize=False
    ):
        """
        Calculate the excess moments of order moment between two dates based on the log-returns from 1-minute period to 60-minute period.

        Parameters:
        - df (pandas.core.frame.DataFrame): Dataframe containing the data of interest. Should contain the following columns:
            'Date': The day of each bar as a string
            'Time Datetime': The time of each bar as a datetime.datetime. Should be with format
                             '%Y-%m-%d %H:%M:%S' (e.g., 2023-11-24 10:00:00)
            'Hour Datetime': The hour of each bar as a datetime.datetime. Should be with format
                             '%Y-%m-%d %H:%M:%S' with %Y-%m-%d=1900-01-01 (e.g., 1900-01-01 10:00:00)
            'Log-returns 1-min Period': The log-returns calculated with 1-min period
            'Log-returns 2-min Period': The log-returns calculated with 2-min period
            ...
            'Log-returns 60-min Period': The log-returns calculated with 60-min period
        - start_date (datetime.datetime): The start date to be considered when calculating the excess moments.
        - end_date (datetime.datetime): The end date to be considered when calculating the excess moments.
        - moment (int, optional): The order of the excess moments to be calculated. Default is 4,
          which means that by default the function returns the excess kurtoses of the returns.
        - filter_minutes (int, optional): The number of minutes to be ignored at the beginning and end of each day. Default is 15.
        - normalize (bool, optional): A boolean value indicating whether the log-returns should be normalized on a daily
          basis or not. Default is False.

        Returns:
        - An numpy.ndarray containing the excess moments for the 60 different periods.
        """
        df_log_returns = df.loc[
            (df["Time Datetime"] > start_date) & (df["Time Datetime"] <= end_date)
        ].copy()
        df_log_returns.reset_index(drop=True, inplace=True)
        list_periods = list(range(1, 61))
        list_excess_moments = []
        for period in list_periods:
            column_name = "Log-returns {}-min Period".format(period)
            df_temp = df_log_returns.loc[df_log_returns[column_name].notnull()][
                ["Date", "Hour Datetime", column_name]
            ].copy()
            df_temp.reset_index(drop=True, inplace=True)
            df_temp.rename(columns={column_name: "Log-returns"}, inplace=True)
            returns = self.compute_log_returns(
                df_temp, period, filter_minutes=filter_minutes, normalize=normalize
            )
            excess_moment = self.calculate_excess_moment(returns, moment=moment)
            list_excess_moments.append(excess_moment)

        return np.array(list_excess_moments).reshape((1, 60))

    @staticmethod
    def plot_moments(
        arr_moment_average_standard,
        arr_moment_one_period_standard,
        arr_moment_average_normalized,
        arr_moment_one_period_normalized,
        start_date,
        end_date,
        frequency,
        moment=4,
        upload_path=None,
    ):
        """
        Plot the excess moments of order moment both with standard returns and normalized returns.

        Parameters:
        - arr_moment_average_standard: The excess moments calculated as average of subperiods using standard returns.
        - arr_moment_one_period_standard: The excess moments calculated over the whole period using standard returns.
        - arr_moment_average_normalized: The excess moments calculated as average of subperiods using normalized returns.
        - arr_moment_one_period_normalized: The excess moments calculated over the whole period using normalized returns.
        - start_date (datetime.datetime): The start date to be considered when calculating the excess moments.
        - end_date (datetime.datetime): The end date to be considered when calculating the excess moments.
        - frequency (int): The number of minutes to be considered for each subperiod.
        - moment (int, optional): The order of the excess moments. Default is 4, which means that
          by default the excess moments are considered to be the excess kurtoses of the returns.
        - upload_path (string or None, optional): The path where to save the figure. If not None, the
          figure is saved according to the input path. If None, the figure is not saved. Default is None.

        Returns:
        - None.
        """
        dict_moment_names = {
            1: "mean",
            2: "variance",
            3: "skewness",
            4: "excess kurtosis",
        }
        title_moment = string.capwords(dict_moment_names[moment])
        legend_moment = dict_moment_names[moment].capitalize()

        nb_days_per_period = str(np.round(frequency / 391, decimals=0)).split(".0")[0]
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        axs[0].plot(
            list(range(1, 61)),
            arr_moment_average_standard,
            label="{} computed as an average of {}-day subperiods".format(
                legend_moment, nb_days_per_period
            ),
        )
        axs[0].plot(
            list(range(1, 61)),
            arr_moment_one_period_standard,
            label="{} computed over the whole period".format(legend_moment),
        )
        axs[0].set_xlabel("Period (in minutes)")
        axs[0].set_ylabel(legend_moment)
        axs[0].axhline(y=0, color="red", linestyle="--")
        axs[0].set_title("{} of Standard Returns".format(title_moment))

        axs[1].plot(list(range(1, 61)), arr_moment_average_normalized)
        axs[1].plot(list(range(1, 61)), arr_moment_one_period_normalized)
        axs[1].set_xlabel("Period (in minutes)")
        axs[1].set_ylabel(legend_moment)
        axs[1].axhline(y=0, color="red", linestyle="--")
        axs[1].set_title("{} of Daily Normalized Returns".format(title_moment))

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.674, 0.87))
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        fig.suptitle(
            "{} as a function of timescale from {} to {}".format(
                legend_moment, start_date, end_date
            ),
            fontsize=22,
            y=0.943,
        )
        fig.subplots_adjust(top=0.68)

        if upload_path is not None:
            plt.savefig(upload_path)
        plt.show()

    def calculate_and_plot_moments(
        self,
        start_date,
        end_date,
        frequency,
        moment=4,
        filter_minutes=15,
        upload_path=None,
    ):
        """
        Calculate and plot the excess moments of order moment both over the whole period and as average of
        subperiods, and both with standard returns and normalized returns. 'Excess' means that the theoretical
        moment of order moment of the standard normal distribution is substracted from the empirical moments.

        Parameters:
        - start_date (datetime.datetime): The start date to be considered when calculating the excess moments.
        - end_date (datetime.datetime): The end date to be considered when calculating the excess moments.
        - frequency (int): The number of minutes to be considered for each subperiod.
        - moment (int, optional): The order of the excess moments to be calculated and plotted. Default is 4,
          which means that by default the function plots the excess kurtoses of the returns.
        - filter_minutes (int, optional): The number of minutes to be ignored at the beginning and end of each day. Default is 15.
        - upload_path (string or None, optional): The path where to save the figure. If not None, the
          figure is saved according to the input path. If None, the figure is not saved. Default is None.

        Returns:
        - None.
        """
        df = self.df.copy()
        start_date = datetime.datetime(
            year=start_date.year,
            month=start_date.month,
            day=start_date.day,
            hour=0,
            minute=0,
        )
        end_date = datetime.datetime(
            year=end_date.year,
            month=end_date.month,
            day=end_date.day,
            hour=23,
            minute=59,
        )
        df = df.loc[
            (df["Time Datetime"] >= start_date) & (df["Time Datetime"] <= end_date)
        ].copy()
        df.reset_index(drop=True, inplace=True)
        start_date = df["Time Datetime"].iloc[0]
        end_date = df["Time Datetime"].iloc[-1]
        start_date = datetime.datetime(
            year=start_date.year,
            month=start_date.month,
            day=start_date.day,
            hour=0,
            minute=0,
        )
        end_date = datetime.datetime(
            year=end_date.year,
            month=end_date.month,
            day=end_date.day,
            hour=23,
            minute=59,
        )

        list_dates = []
        i = df.loc[df["Time Datetime"] <= end_date].index[-1]
        current_date = end_date
        while current_date >= start_date:
            list_dates.append(current_date)
            i -= frequency
            if i < 0:
                break
            current_date = df.iloc[i]["Time Datetime"]
            current_date = df.loc[df["Date"] == current_date.strftime("%Y-%m-%d")].iloc[
                0
            ]["Time Datetime"]
            i = df.loc[df["Date"] == current_date.strftime("%Y-%m-%d")].iloc[0].name

        arr_moment_full_standard = np.empty((0, 60))
        arr_moment_full_normalized = np.empty((0, 60))
        for i in range(len(list_dates) - 1):
            start_date_temp = list_dates[i + 1]
            end_date_temp = list_dates[i]
            arr_moment_standard = self.compute_moment_between_two_dates(
                df,
                start_date_temp,
                end_date_temp,
                moment=moment,
                filter_minutes=filter_minutes,
                normalize=False,
            )
            arr_moment_full_standard = np.concatenate(
                [arr_moment_full_standard, arr_moment_standard], axis=0
            )
            arr_moment_normalized = self.compute_moment_between_two_dates(
                df,
                start_date_temp,
                end_date_temp,
                moment=moment,
                filter_minutes=filter_minutes,
                normalize=True,
            )
            arr_moment_full_normalized = np.concatenate(
                [arr_moment_full_normalized, arr_moment_normalized], axis=0
            )

        arr_moment_average_standard = np.mean(arr_moment_full_standard, axis=0)
        arr_moment_average_normalized = np.mean(arr_moment_full_normalized, axis=0)
        arr_moment_one_period_standard = self.compute_moment_between_two_dates(
            df,
            start_date,
            end_date,
            moment=moment,
            filter_minutes=filter_minutes,
            normalize=False,
        ).reshape(60)
        arr_moment_one_period_normalized = self.compute_moment_between_two_dates(
            df,
            start_date,
            end_date,
            moment=moment,
            filter_minutes=filter_minutes,
            normalize=True,
        ).reshape(60)

        self.plot_moments(
            arr_moment_average_standard,
            arr_moment_one_period_standard,
            arr_moment_average_normalized,
            arr_moment_one_period_normalized,
            start_date,
            end_date,
            frequency,
            moment=moment,
            upload_path=upload_path,
        )
