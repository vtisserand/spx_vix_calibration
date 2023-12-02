import pandas as pd
import datetime
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import statsmodels.api as sm
from statsmodels.tools.validation import (
    bool_like,
    float_like,
    int_like
)
from statsmodels.tools.sm_exceptions import ValueWarning
from scipy.linalg import toeplitz
from scipy import stats
from statsmodels.compat.python import lzip


class DataFrameTools:

    @staticmethod
    def construct_datetime_columns(df, construct_time=True, construct_date=True, construct_hour=True):
        if construct_time:
            df['Time Datetime'] = df['Time'].apply(lambda value: datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S"))
        if construct_date:
            df['Date'] = df['Time'].apply(lambda value: value.split(' ')[0])
        if construct_hour:
            df['Hour Datetime'] = df['Hour'].apply(lambda value: datetime.datetime.strptime(value, '%H:%M:%S'))
        return df
    



class AutocorrelationTools:

    def __init__(self, df):
        self.df = df.copy()

    @staticmethod
    def acf_pearson(returns, lag, adjust_denominator=False, adjust_daily=False):

        n = len(returns)
        mean_returns = np.mean(returns)
        variance_returns = np.var(returns)
        returns = returns - mean_returns
        
        if lag == 0:
            arr_corr_left = returns
            arr_corr_right = arr_corr_left
        else:
            df = returns.iloc[:-lag].reset_index()
            df_lagged = returns.iloc[lag:].reset_index().rename(columns={'Time': 'Time Lagged', 'Log-returns': 'Log-returns Lagged'})
            df = pd.concat([df, df_lagged], axis=1)
            if adjust_daily:
                df = df.loc[df.apply(lambda row: row['Time'].split(' ')[0] == row['Time Lagged'].split(' ')[0], axis=1)].copy()
            arr_corr_left = np.array(df['Log-returns'].to_list())
            arr_corr_right = np.array(df['Log-returns Lagged'].to_list())

        print('Lag ACF: {}'.format(lag))

        autocovariance = (1 / (n - lag * adjust_denominator)) * np.correlate(arr_corr_left, arr_corr_right)
        autocorrelation = autocovariance / variance_returns
        return autocorrelation[0]


    def acf(self, returns, nlags=20, alpha=None, adjust_denominator=False, adjust_daily=False):

        nlags = int_like(nlags, "nlags", optional=True)
        alpha = float_like(alpha, "alpha", optional=True)
        adjust_daily = bool_like(adjust_daily, "adjust_daily", optional=True)

        acf = [1.0]
        for k in range(1, nlags + 1):
            acf.append(self.acf_pearson(returns, k, adjust_denominator=adjust_denominator, adjust_daily=adjust_daily))
        ret = np.array(acf)
        
        if alpha is not None:
            varacf = 1.0 / len(returns)
            confint = stats.norm.ppf(1.0 - ((1 - alpha) / 2.0)) * np.sqrt(varacf)
            return ret, confint
        else:
            return ret


    @staticmethod
    def pacf_yule_walker(returns, order=1, adjust_denominator=False, adjust_daily=False):
        """
        Estimate AR(p) parameters from a sequence using the Yule-Walker equations.

        Adjusted or maximum-likelihood estimator (mle)

        Parameters
        ----------
        x : array_like
            A 1d array.
        order : int, optional
            The order of the autoregressive process. Default is 1.
        adjust_denominator : bool, optional
        Determines denominator in estimate of autocorrelation function (ACF) at
        lag k. If False, the denominator is n=len(returns), if True
        the denominator is n-k. The default is False.

        Returns
        -------
        rho : ndarray
            AR(p) coefficients computed using the Yule-Walker method.
        sigma : float
            The estimate of the residual standard deviation.
        """
        returns -= returns.mean()
        n = len(returns)

        r = np.zeros(order+1, np.float64)
        r[0] = returns.var(ddof=0)
        
        for k in range(1, order+1):
            df = returns.iloc[:-k].reset_index()
            df_lagged = returns.iloc[k:].reset_index().rename(columns={'Time': 'Time Lagged', 'Log-returns': 'Log-returns Lagged'})
            df = pd.concat([df, df_lagged], axis=1)
            if adjust_daily:
                df = df.loc[df.apply(lambda row: row['Time'].split(' ')[0] == row['Time Lagged'].split(' ')[0], axis=1)].copy()
            arr_corr_left = np.array(df['Log-returns'].to_list())
            arr_corr_right = np.array(df['Log-returns Lagged'].to_list())
            r[k] = (1 / (n - k * adjust_denominator)) * np.correlate(arr_corr_left, arr_corr_right)[0]

        print(r)
        R = toeplitz(r[:-1])

        try:
            rho = np.linalg.solve(R, r[1:])
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                warnings.warn("Matrix is singular. Using pinv.", ValueWarning)
                rho = np.linalg.pinv(R) @ r[1:]
            else:
                raise

        sigmasq = r[0] - (r[1:]*rho).sum()
        if not np.isnan(sigmasq) and sigmasq > 0:
            sigma = np.sqrt(sigmasq)
        else:
            sigma = np.nan
        return rho, sigma


    def pacf(self, returns, nlags=20, alpha=None, adjust_denominator=False, adjust_daily=False):
        """
        Partial autocorrelation estimate using method ywadjusted (non-recursive yule_walker, i.e. Yule-Walker with sample-size adjustment in
        denominator for acovf).

        Parameters
        ----------
        returns (pandas.Series): Observations of time series for which pacf is calculated.
        nlags (int): Number of lags to return autocorrelation for. Default is 20.
        alpha (float): If a number is given, the confidence intervals for the given level are
                    returned. For instance if alpha=.05, 95 % confidence intervals are
                    returned where the standard deviation is computed according to 1/sqrt(len(x)).
        adjust_daily (bool, optional): If True, the autocovariances used in estimating the partial
                    autocorrelations are computed by multiplying returns on same days only.

        Returns
        -------
        pacf : ndarray
            The partial autocorrelations for lags 0, 1, ..., nlags. Shape
            (nlags+1,).
        confint : float, optional
            Confidence intervals for the PACF. Returned if alpha is not None.
        """
        nlags = int_like(nlags, "nlags", optional=True)
        alpha = float_like(alpha, "alpha", optional=True)
        adjust_daily = bool_like(adjust_daily, "adjust_daily", optional=True)

        pacf = [1.0]
        for k in range(1, nlags + 1):
            pacf.append(self.pacf_yule_walker(returns, k, adjust_denominator=adjust_denominator, adjust_daily=adjust_daily)[0][-1])
        ret = np.array(pacf)

        if alpha is not None:
            varacf = 1.0 / len(returns)
            confint = stats.norm.ppf(1.0 - ((1 - alpha) / 2.0)) * np.sqrt(varacf)
            return ret, confint
        else:
            return ret


    def plot_autocorrelations_without_pacf(self, returns, nlags, alpha=0.05, adjust_denominator=False,
                                           adjust_daily=False, title="Autocorrelations"):
        """
        Plot the ACF of the time series given as input.

        Parameters:
        - returns (list or numpy.ndarray): Time series
        - nlags (int): The maximum lag to be displayed
        - confidence_level (float): The confidence level for the confidence interval of the autocorrelations
        - title (string): The title of the figure.

        Returns:
        - None
        """
        arr_acf, confidence_band = self.acf(returns, nlags=nlags, alpha=alpha,
                                            adjust_denominator=adjust_denominator, adjust_daily=adjust_daily)
        str_confidence_level = str(np.round(100 * alpha, decimals=0)).split('.0')[0]

        arr_acf = arr_acf[1:]
        x = np.array(range(1, nlags+1))

        fig, ax = plt.subplots(figsize=(7, 4))
        color = '#1f77b4'
        margin = 0.01
        max_abs_value = max(abs(min(arr_acf)), abs(max(arr_acf)))
        max_y_value = max_abs_value + margin
        min_y_value = -max_abs_value - margin

        # Plot ACF using stem plot
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        markerline, stemlines, baseline = ax.stem(x, arr_acf, linefmt='-', basefmt=' ')
        plt.setp(stemlines, 'color', color) # Set stem line color
        plt.setp(markerline, 'color', color) # Set marker line color
        ax.set_xlabel('Lag (in minutes)')
        ax.set_ylabel('ACF')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(min_y_value, max_y_value)
        odd_x_ticks = np.arange(1, nlags + 1, 2)
        ax.set_xticks(odd_x_ticks)
        ax.set_xticklabels(odd_x_ticks)
        ax.axhline(y=confidence_band, color='red', linestyle='--', label="{}% confidence interval".format(str_confidence_level))
        ax.axhline(y=-confidence_band, color='red', linestyle='--')

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.68, 0.88))
        fig.suptitle(title, fontsize=22, y=0.97)
        fig.subplots_adjust(hspace=0.2, top=0.75)
        plt.show()
    

    def plot_autocorrelations_with_pacf(self, arr_acf, arr_pacf, confidence_band, nlags, alpha=0.05, title="Autocorrelations"):
        """
        Plot the ACF and PACF of the time series given as input.

        Parameters:
        - returns (list or numpy.ndarray): Time series
        - nlags (int): The maximum lag to be displayed
        - confidence_level (float): The confidence level for the confidence interval of the autocorrelations
        - title (string): The title of the figure.

        Returns:
        - None
        """
        str_confidence_level = str(np.round(100 * alpha, decimals=0)).split('.0')[0]

        arr_acf = arr_acf[1:]
        arr_pacf = arr_pacf[1:]
        x = np.array(range(1, nlags+1))

        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(7, 4))
        margin = 0.01
        max_abs_value = max(abs(min(min(arr_acf), min(arr_pacf))), abs(max(max(arr_acf), max(arr_pacf))))
        max_y_value = max_abs_value + margin
        min_y_value = -max_abs_value - margin

        color = '#1f77b4'
        color_stemlines = color
        color_markerline = color

        # Plot ACF using stem plot
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        if nlags <= 2:
            markerline, stemlines, baseline = ax1.stem(x, arr_acf, linefmt='-', basefmt=' ')
            plt.setp(stemlines, 'color', color_stemlines) # Set stem line color
            plt.setp(markerline, 'color', color_markerline) # Set marker line color
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            odd_x_ticks = np.arange(1, nlags + 1, 2)
            ax1.set_xticks(odd_x_ticks)
            ax1.set_xticklabels(odd_x_ticks)
        else:
            ax1.scatter(x, arr_acf, color=color, marker='o', s=10)
        ax1.set_ylabel('ACF')
        ax1.set_ylim(min_y_value, max_y_value)
        ax1.axhline(y=confidence_band, color='red', linestyle='--', label="{}% confidence interval".format(str_confidence_level))
        ax1.axhline(y=-confidence_band, color='red', linestyle='--')

        # Plot PACF using stem plot
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        if nlags <= 2:
            markerline, stemlines, baseline = ax2.stem(x, arr_pacf, linefmt='-', basefmt=' ')
            plt.setp(stemlines, 'color', color_stemlines) # Set stem line color
            plt.setp(markerline, 'color', color_markerline) # Set marker line color
        else:
            ax2.scatter(x, arr_pacf, color=color, marker='o', s=10)
        ax2.set_xlabel('Lag (in minutes)')
        ax2.set_ylabel('PACF')
        ax2.set_ylim(min_y_value, max_y_value)
        ax2.axhline(y=confidence_band, color='red', linestyle='--')
        ax2.axhline(y=-confidence_band, color='red', linestyle='--')

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.9, 0.985))
        fig.suptitle(title, fontsize=22, x=0.35, y=0.97)
        fig.subplots_adjust(hspace=0.2, top=0.85)
        
        # Plot intraday autocorrelations, i.e. autocorrelations where the products are only with returns
        # on a same day. This means that we can't have a lag larger than 330, and that the larger
        # the lag, the less data we use to compute the autocorrelations -> Thus, the confidence bands
        # needs to be adjusted.

        # Modifier la fonction plot_autocorrelations_without_pacf
        
        #plt.savefig('Absolute_autocorrelations.pdf')
        plt.show()


    def plot_autocorrelations(self, start_date, end_date, with_pacf, nlags=20, alpha=0.95, adjust_denominator=False,
        adjust_daily=False, transformation=lambda x: x, title="Autocorrelations"):
        """
        Construct the series of returns and plot the autocorrelations

        Parameters:
        - start_date (datetime.datetime): The start date to be considered when calculating the autocorrelations
        - end_date (datetime.datetime): The end date to be considered when calculating the autocorrelations
        - with_pacf (bool): A boolean value equal to True if the PACF needs to be displayed, otherwise False
        - nlags (int): The maximum lag to be displayed. Default is 20
        - confidence_level (float): The confidence level for the confidence interval of the autocorrelations. Default is 0.95
        - transformation (func): A function to be applied to the returns, e.g. the absolute value function. Default is the identity function.
        - title (string): The title of the figure. Default is 'Autocorrelations'

        Returns:
        - None
        """
        df = self.df.copy()
        df = df.loc[(df['Time Datetime'] >= start_date) & (df['Time Datetime'] <= end_date)].copy()
        df.reset_index(drop=True, inplace=True)

        daily_start_date = datetime.datetime(year=1900, month=1, day=1, hour=10, minute=0)
        daily_end_date = datetime.datetime(year=1900, month=1, day=1, hour=15, minute=30)
        df = df.loc[(df['Hour Datetime'] >= daily_start_date) & (df['Hour Datetime'] <= daily_end_date)].copy()

        df.set_index('Time', inplace=True)
        returns = df['Log-returns 1-min Period']
        returns.name = 'Log-returns'
        returns = transformation(returns)
        if with_pacf:
            self.plot_autocorrelations_with_pacf(returns, nlags=nlags, alpha=alpha,
                adjust_denominator=adjust_denominator, adjust_daily=adjust_daily, title=title)
        else:
            self.plot_autocorrelations_without_pacf(returns, nlags=nlags, alpha=alpha,
                adjust_denominator=adjust_denominator, adjust_daily=adjust_daily, title=title)








    @staticmethod
    def pacf_yule_walker_test(returns, order=1, adjust_denominator=False, adjust_daily=False):
        """
        Estimate AR(p) parameters from a sequence using the Yule-Walker equations.

        Adjusted or maximum-likelihood estimator (mle)

        Parameters
        ----------
        x : array_like
            A 1d array.
        order : int, optional
            The order of the autoregressive process. Default is 1.
        adjust_denominator : bool, optional
        Determines denominator in estimate of autocorrelation function (ACF) at
        lag k. If False, the denominator is n=len(returns), if True
        the denominator is n-k. The default is False.

        Returns
        -------
        rho : ndarray
            AR(p) coefficients computed using the Yule-Walker method.
        sigma : float
            The estimate of the residual standard deviation.
        """
        returns -= returns.mean()
        n = len(returns)

        r = np.zeros(order+1, np.float64)
        r[0] = returns.var(ddof=0)
        
        for k in range(1, order+1):
            df = returns.iloc[:-k].reset_index()
            df_lagged = returns.iloc[k:].reset_index().rename(columns={'Time': 'Time Lagged', 'Log-returns': 'Log-returns Lagged'})
            df = pd.concat([df, df_lagged], axis=1)
            if adjust_daily:
                df = df.loc[df.apply(lambda row: row['Time'].split(' ')[0] == row['Time Lagged'].split(' ')[0], axis=1)].copy()
            arr_corr_left = np.array(df['Log-returns'].to_list())
            arr_corr_right = np.array(df['Log-returns Lagged'].to_list())
            r[k] = (1 / (n - k * adjust_denominator)) * np.correlate(arr_corr_left, arr_corr_right)[0]
            if k == order:
                print('Lag PACF: {}, len={}'.format(k, len(df)))













    def pacf_fast(self, r):
        """
        Partial autocorrelation estimate using method ywadjusted (non-recursive yule_walker, i.e. Yule-Walker with sample-size adjustment in
        denominator for acovf).

        Parameters
        ----------
        returns (pandas.Series): Observations of time series for which pacf is calculated.
        nlags (int): Number of lags to return autocorrelation for. Default is 20.
        alpha (float): If a number is given, the confidence intervals for the given level are
                    returned. For instance if alpha=.05, 95 % confidence intervals are
                    returned where the standard deviation is computed according to 1/sqrt(len(x)).
        adjust_daily (bool, optional): If True, the autocovariances used in estimating the partial
                    autocorrelations are computed by multiplying returns on same days only.

        Returns
        -------
        pacf : ndarray
            The partial autocorrelations for lags 0, 1, ..., nlags. Shape
            (nlags+1,).
        confint : float, optional
            Confidence intervals for the PACF. Returned if alpha is not None.
        """
        nlags = len(r)

        pacf = [1.0]
        for k in range(1, nlags):
            r_temp = r[:k + 1]
            R = toeplitz(r_temp[:-1])
            try:
                rho = np.linalg.solve(R, r_temp[1:])
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    warnings.warn("Matrix is singular. Using pinv.", ValueWarning)
                    rho = np.linalg.pinv(R) @ r_temp[1:]
                else:
                    raise
            pacf.append(rho[-1])
        
        return np.array(pacf)



    def plot_autocorrelations_fast(self, returns, nlags=20, alpha=0.95, adjust_denominator=False,
        adjust_daily=False):
        """
        Construct the series of returns and plot the autocorrelations

        Parameters:
        - start_date (datetime.datetime): The start date to be considered when calculating the autocorrelations
        - end_date (datetime.datetime): The end date to be considered when calculating the autocorrelations
        - with_pacf (bool): A boolean value equal to True if the PACF needs to be displayed, otherwise False
        - nlags (int): The maximum lag to be displayed. Default is 20
        - confidence_level (float): The confidence level for the confidence interval of the autocorrelations. Default is 0.95
        - transformation (func): A function to be applied to the returns, e.g. the absolute value function. Default is the identity function.
        - title (string): The title of the figure. Default is 'Autocorrelations'

        Returns:
        - None
        """

        varacf = 1.0 / len(returns)
        confint = stats.norm.ppf(1.0 - ((1 - alpha) / 2.0)) * np.sqrt(varacf)


        variance_returns = returns.var(ddof=0)
        returns -= returns.mean()
        n = len(returns)

        acf = np.zeros(nlags + 1, np.float64)
        acf[0] = 1
        r = np.zeros(nlags + 1, np.float64)
        r[0] = variance_returns
        for k in range(1, nlags + 1):
            df = returns.iloc[:-k].reset_index()
            df_lagged = returns.iloc[k:].reset_index().rename(columns={'Time': 'Time Lagged', 'Log-returns': 'Log-returns Lagged'})
            df = pd.concat([df, df_lagged], axis=1)
            if adjust_daily:
                df = df.loc[df.apply(lambda row: row['Time'].split(' ')[0] == row['Time Lagged'].split(' ')[0], axis=1)].copy()
            arr_corr_left = np.array(df['Log-returns'].to_list())
            arr_corr_right = np.array(df['Log-returns Lagged'].to_list())
            r[k] = (1 / (n - k * adjust_denominator)) * np.correlate(arr_corr_left, arr_corr_right)[0]
            acf[k] = r[k] / variance_returns


            if k == nlags:
                print('Lag PACF: {}, len={}'.format(k, len(df)))
        
        pacf = self.pacf_fast(r)
        return acf, pacf, confint

    

            


class KurtosisTools:

    def __init__(self, df):
        self.df = df.copy()


    @staticmethod
    def compute_log_returns(df, normalize):
        """
        Adjust the log-returns contained in the dataframe df, so as to exclude the first log-return of each day. Indeed, we
        look at time series limited to trading days from 9:30am to 4:00pm, so that the first log-return is computed without
        taking into account 'after-hours' trading activity. See https://arxiv.org/abs/2311.07738 for a similar methodology.

        Parameters:
        - df (pandas.core.frame.DataFrame): Dataframe containing the data of interest. Should contain the following columns:
            'Date': The day of each bar as a string
            'Log-returns': The log-returns
        - normalize (bool): A boolean value indicating whether the log-returns should be normalized on a daily basis or not

        Returns:
        - An numpy.ndarray containing treated log-returns
        """
        # Removing the first log-return of each day (since it was calculated using the price of the previous day)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Index'}, inplace=True)
        list_indices = df.groupby('Date').first()['Index'].to_list()
        df.drop(index=list_indices, inplace=True)
        df.drop('Index', axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        if normalize:
            dict_daily_averages = df.groupby('Date').mean().to_dict()['Log-returns']
            dict_daily_sd = df.groupby('Date').std().to_dict()['Log-returns']
            return np.array(df.apply(lambda row: (row['Log-returns'] - dict_daily_averages[row['Date']]) / dict_daily_sd[row['Date']], axis=1).to_list())
        else:
            return np.array(df['Log-returns'].to_list())


    @staticmethod
    def calculate_excess_kurtosis(arr_returns):
        """
        Calculate excess kurtosis
        """
        mean_returns = np.mean(arr_returns)
        std_returns = np.std(arr_returns)
        
        # Calculate kurtosis
        numerator = np.mean((arr_returns - mean_returns) ** 4)
        denominator = std_returns ** 4
        
        kurtosis = numerator / denominator - 3
        return kurtosis


    def compute_kurtosis_between_two_dates(self, start_date, end_date, normalize):
        """
        Calculate the kurtosis between two dates based on the log-returns from 1-minute period to 60-minute period.

        Parameters:
        - start_date (datetime.datetime): The start date to be considered when calculating the kurtosis
        - end_date (datetime.datetime): The end date to be considered when calculating the kurtosis
        - normalize (bool): A boolean value indicating whether the log-returns should be normalized on a daily basis or not

        Returns:
        - An numpy.ndarray containing the kurtosis for the 60 different periods
        """
        df_log_returns = self.df.loc[(self.df['Time Datetime'] > start_date) & (self.df['Time Datetime'] <= end_date)].copy()
        df_log_returns.reset_index(drop=True, inplace=True)
        list_periods = list(range(1, 61))
        list_excess_kurtosis = []
        for period in list_periods:
            column_name = 'Log-returns {}-min Period'.format(period)
            df_temp = df_log_returns.loc[df_log_returns[column_name].notnull()][['Date', column_name]].copy()
            df_temp.reset_index(drop=True, inplace=True)
            df_temp.rename(columns={column_name: 'Log-returns'}, inplace=True)
            arr_returns = self.compute_log_returns(df_temp, normalize)
            kurtosis = self.calculate_excess_kurtosis(arr_returns)
            list_excess_kurtosis.append(kurtosis)
        
        return np.array(list_excess_kurtosis).reshape((1, 60))


    def compute_kurtosis_to_plot(self, start_date, end_date, frequency, normalize):
        """
        Calculate the kurtosis over the whole period and as average of subperiods using the method self.compute_kurtosis_between_two_dates

        Parameters:
        - start_date (datetime.datetime): The start date to be considered when calculating the kurtosis
        - end_date (datetime.datetime): The end date to be considered when calculating the kurtosis
        - frequency (int): The number of minutes to be considered for each subperiod
        - normalize (bool): A boolean value indicating whether the log-returns should be normalized on a daily basis or not

        Returns:
        - arr_kurtosis_average: A numpy.ndarray containing the kurtosis calculated as average of subperiods
        - arr_kurtosis_one_period: A numpy.ndarray containing the kurtosis calculated over the whole period
        """
        list_dates = []
        
        i = self.df.loc[self.df['Time Datetime'] <= end_date].index[-1]
        current_date = end_date
        while current_date >= start_date:
            list_dates.append(current_date)
            i -= frequency
            if i < 0:
                break
            current_date = self.df.iloc[i]['Time Datetime']
            current_date = self.df.loc[self.df['Date'] == current_date.strftime('%Y-%m-%d')].iloc[0]['Time Datetime']
            i = self.df.loc[self.df['Date'] == current_date.strftime('%Y-%m-%d')].iloc[0].name
        
        arr_kurtosis_full = np.empty((0, 60))
        for i in range(len(list_dates) - 1):
            start_date_temp = list_dates[i+1]
            end_date_temp = list_dates[i]
            arr_kurtosis = self.compute_kurtosis_between_two_dates(start_date_temp, end_date_temp, normalize)
            arr_kurtosis_full = np.concatenate([arr_kurtosis_full, arr_kurtosis], axis=0)
        
        arr_kurtosis_average = np.mean(arr_kurtosis_full, axis=0)
        arr_kurtosis_one_period = self.compute_kurtosis_between_two_dates(start_date, end_date, normalize).reshape(60)
        return arr_kurtosis_average, arr_kurtosis_one_period


    @staticmethod
    def plot_kurtosis(arr_kurtosis_average_standard,
                      arr_kurtosis_one_period_standard,
                      arr_kurtosis_average_normalized,
                      arr_kurtosis_one_period_normalized,
                      start_date,
                      end_date,
                      frequency):
        """
        Plot the kurtosis both with standard returns and normalized returns

        Parameters:
        - arr_kurtosis_average_standard: The kurtosis calculated as average of subperiods using standard returns
        - arr_kurtosis_one_period_standard: The kurtosis calculated over the whole period using standard returns
        - arr_kurtosis_average_normalized: The kurtosis calculated as average of subperiods using normalized returns
        - arr_kurtosis_one_period_normalized: The kurtosis calculated over the whole period using normalized returns
        - start_date (datetime.datetime): The start date to be considered when calculating the kurtosis
        - end_date (datetime.datetime): The end date to be considered when calculating the kurtosis
        - frequency (int): The number of minutes to be considered for each subperiod

        Returns:
        - None
        """
        nb_days_per_period = str(np.round(frequency / 391, decimals=0)).split('.0')[0]
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        
        axs[0].plot(list(range(1, 61)), arr_kurtosis_average_standard, label='Kurtosis computed as an average of {}-day subperiods'.format(nb_days_per_period))
        axs[0].plot(list(range(1, 61)), arr_kurtosis_one_period_standard, label='Kurtosis computed over the whole period')
        axs[0].set_xlabel('Period (in minutes)')
        axs[0].set_ylabel('Excess kurtosis')
        axs[0].axhline(y=0, color='red', linestyle='--')
        axs[0].set_title("Excess Kurtosis of Standard Returns")
        
        axs[1].plot(list(range(1, 61)), arr_kurtosis_average_normalized)
        axs[1].plot(list(range(1, 61)), arr_kurtosis_one_period_normalized)
        axs[1].set_xlabel('Period (in minutes)')
        axs[1].set_ylabel('Excess kurtosis')
        axs[1].axhline(y=0, color='red', linestyle='--')
        axs[1].set_title("Excess Kurtosis of Daily Normalized Returns")
        
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.648, 0.87))
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        fig.suptitle("Kurtosis as a function of timescale from {} to {}".format(start_date, end_date), fontsize=22, y=0.943)
        fig.subplots_adjust(top=0.68)
        plt.show()


    def calculate_and_plot_kurtosis(self, start_date, end_date, frequency):
        """
        Calculate and plot the kurtosis both with standard returns and normalized returns

        Parameters:
        - start_date (datetime.datetime): The start date to be considered when calculating the kurtosis
        - end_date (datetime.datetime): The end date to be considered when calculating the kurtosis
        - frequency (int): The number of minutes to be considered for each subperiod

        Returns:
        - None
        """
        arr_kurtosis_average_standard, arr_kurtosis_one_period_standard = self.compute_kurtosis_to_plot(start_date, end_date, frequency, False)
        arr_kurtosis_average_normalized, arr_kurtosis_one_period_normalized = self.compute_kurtosis_to_plot(start_date, end_date, frequency, True)
        self.plot_kurtosis(arr_kurtosis_average_standard,
                           arr_kurtosis_one_period_standard,
                           arr_kurtosis_average_normalized,
                           arr_kurtosis_one_period_normalized,
                           start_date,
                           end_date,
                           frequency)






