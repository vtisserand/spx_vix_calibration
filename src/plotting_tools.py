import pandas as pd
import datetime
import numpy as np
import scipy
import string
import dateutil
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.tools.sm_exceptions import ValueWarning


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
    def pacf_yule_walker(r):
        """
        Compute the partial autocorrelation estimates using Yule Walker equations.

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
            print('Yule-Walker lag: {}'.format(k))
            r_temp = r[:k + 1]
            R = scipy.linalg.toeplitz(r_temp[:-1])
            try:
                rho = np.linalg.solve(R, r_temp[1:])
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    warnings.warn("Matrix is singular. Using pinv.", ValueWarning)
                    rho = np.linalg.pinv(R) @ r_temp[1:]
                else:
                    raise
            pacf.append(rho[-1])
        pacf = np.array(pacf)
    
        return pacf


    def calculate_autocorrelations(self, returns, nlags=20, alpha=0.05, adjust_denominator=False, adjust_daily=False):
        """
        Compute the autocorrelations and partial autocorrelations.

        Parameters
        ----------
        returns (pandas.core.series.Series): A pandas Series containing the returns.
            The index should be named 'Time' and contain the times associated with the returns as strings
            with format '%Y-%m-%d %H:%M:%S', while the Series itself should be named 'Log-returns'.
        nlags (int, optional): The number of lags to return autocorrelation for. Default is 20.
        alpha (float, optional): The confidence level for the confidence intervals of the autocorrelations.
            For instance if alpha=.05, 95% confidence intervals are returned where the standard deviation
            is computed according to 1/sqrt(number of observations). Default is 0.05.
        adjust_denominator (bool, optional): Determines denominator in estimate of autocorrelation function (ACF)
            at lag k. If False, the denominator is n=len(returns), if True the denominator is n-k. Default is False.
        adjust_daily (bool, optional): If True, the autocovariances used in estimating the
            autocorrelations are computed by multiplying returns on same days only. This means that
            the larger the lag, the less data we use to compute the autocorrelations. Default is False.

        Returns
        -------
        acf (numpy.ndarray): The autocorrelations for lags 0, 1, ..., nlags computed using
            Pearson autocorrelation coefficients. Shape (nlags+1,).
        pacf (numpy.ndarray): The partial autocorrelations for lags 0, 1, ..., nlags
            computed using Yule Walker equations. Shape (nlags+1,).
        confint (numpy.ndarray): The upper bounds of the symmetric confidence intervals for the
            autocorrelations of lags 0, 1, ..., nlags. That is, the confidence interval of
            autocorrelation of lag k is given by [-confint[k], confint[k]]. Shape (nlags+1,).
        """
        variance_returns = returns.var(ddof=0)
        returns -= returns.mean()
        coef = scipy.stats.norm.ppf(1.0 - (alpha / 2.0))

        acf = np.zeros(nlags + 1, np.float64)
        acf[0] = 1
        r = np.zeros(nlags + 1, np.float64)
        r[0] = variance_returns
        confint = np.zeros(nlags + 1, np.float64)
        confint[0] = 1
        for k in range(1, nlags + 1):
            df = returns.iloc[:-k].reset_index()
            df_lagged = returns.iloc[k:].reset_index().rename(columns={'Time': 'Time Lagged', 'Log-returns': 'Log-returns Lagged'})
            df = pd.concat([df, df_lagged], axis=1)
            if adjust_daily:
                df = df.loc[df.apply(lambda row: row['Time'].split(' ')[0] == row['Time Lagged'].split(' ')[0], axis=1)].copy()
            arr_corr_left = np.array(df['Log-returns'].to_list())
            arr_corr_right = np.array(df['Log-returns Lagged'].to_list())
            n = len(arr_corr_left) + k
            confint[k] = coef * np.sqrt(1.0 / n)
            r[k] = (1 / (n - k * adjust_denominator)) * np.correlate(arr_corr_left, arr_corr_right)[0]
            acf[k] = r[k] / variance_returns
            print('Construction lag: {}, nb datapoints={}'.format(k, n))
        
        print('__________________')
        
        pacf = self.pacf_yule_walker(r)
        return acf, pacf, confint


    def plot_acf_pacf(self, acf, pacf, confint, start_date, end_date, alpha, upload_path=None):
        """
        Plot the ACF and PACF of the time series.

        Parameters
        ----------
        acf (numpy.ndarray): The autocorrelations for lags 0, 1, ..., len(acf)-1.
        pacf (numpy.ndarray): The partial autocorrelations for lags 0, 1, ..., len(pacf)-1.
        confint (numpy.ndarray): The upper bounds of the symmetric confidence intervals for the
            autocorrelations of lags 0, 1, ..., len(confint)-1. That is, the confidence interval of
            autocorrelation of lag k should be given by [-confint[k], confint[k]].
        start_date (datetime.datetime): The start date to be considered when calculating the autocorrelations.
        end_date (datetime.datetime): The end date to be considered when calculating the autocorrelations.
        alpha (float): The confidence level for the confidence intervals of the autocorrelations.
        upload_path (string or None, optional): The path where to save the figure. If not None, the figure
            is saved according to the input path. If None, the figure is not saved. Default is None.

        Returns
        -------
        None.
        """
        str_confidence_level = str(np.round(100 * (1 - alpha), decimals=0)).split('.0')[0]

        acf = acf[1:]
        pacf = pacf[1:]
        confint = confint[1:]
        nlags = len(acf)
        x = np.array(range(1, nlags+1))

        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(7, 4))
        max_nlags_stem = 50
        x_margin = max(0.5, (len(acf) + 1) / 60)
        y_margin = 0.01
        max_abs_value = max(abs(min(min(acf), min(pacf))), abs(max(max(acf), max(pacf))))
        max_y_value = max_abs_value + y_margin
        min_y_value = -max_abs_value - y_margin

        color = '#1f77b4'
        color_stemlines = color
        color_markerline = color

        # Plot ACF using stem plot
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        if nlags <= max_nlags_stem:
            markerline, stemlines, baseline = ax1.stem(x, acf, linefmt='-', basefmt=' ')
            plt.setp(stemlines, 'color', color_stemlines) # Set stem line color
            plt.setp(markerline, 'color', color_markerline) # Set marker line color
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            odd_x_ticks = np.arange(1, nlags + 1, 2)
            ax1.set_xticks(odd_x_ticks)
            ax1.set_xticklabels(odd_x_ticks)
        else:
            ax1.scatter(x, acf, color=color, marker='o', s=10)
        ax1.set_xlim(1 - x_margin, max(x) + x_margin)
        ax1.set_ylabel('ACF')
        ax1.set_ylim(min_y_value, max_y_value)
        ax1.plot(x, confint, color='red', linestyle='--', label="{}% confidence interval".format(str_confidence_level))
        ax1.plot(x, -confint, color='red', linestyle='--')

        # Plot PACF using stem plot
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        if nlags <= max_nlags_stem:
            markerline, stemlines, baseline = ax2.stem(x, pacf, linefmt='-', basefmt=' ')
            plt.setp(stemlines, 'color', color_stemlines) # Set stem line color
            plt.setp(markerline, 'color', color_markerline) # Set marker line color
        else:
            ax2.scatter(x, pacf, color=color, marker='o', s=10)
        ax2.set_xlabel('Lag (in minutes)')
        ax2.set_xlim(1 - x_margin, max(x) + x_margin)
        ax2.set_ylabel('PACF')
        ax2.set_ylim(min_y_value, max_y_value)
        ax2.plot(x, confint, color='red', linestyle='--')
        ax2.plot(x, -confint, color='red', linestyle='--')

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.68, 0.874))
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        fig.suptitle("Autocorrelations from {} to {}".format(start_date, end_date), fontsize=15, y=0.943)
        fig.subplots_adjust(hspace=0.2, top=0.75)

        if upload_path is not None:
            plt.savefig(upload_path)
        plt.show()


    def plot_autocorrelations(self, start_date, end_date, filter_minutes=15, nlags=20, alpha=0.05, adjust_denominator=False,
        adjust_daily=False, transformation=lambda x: x, upload_path=None):
        """
        Construct the series of returns and plot the autocorrelations.

        Parameters
        ----------
        start_date (datetime.datetime): The start date to be considered when calculating the autocorrelations.
        end_date (datetime.datetime): The end date to be considered when calculating the autocorrelations.
        filter_minutes (int, optional): The number of minutes to be ignored at the beginning and end of each day. Default is 15
        nlags (int, optional): The number of lags to plot autocorrelation for. Default is 20.
        alpha (float, optional): The confidence level for the confidence intervals of the autocorrelations.
            For instance if alpha=.05, 95% confidence intervals are plotted where the standard deviation
            is computed according to 1/sqrt(number of observations). Default is 0.05.
        adjust_denominator (bool, optional): Determines denominator in estimate of autocorrelation function (ACF)
            at lag k. If False, the denominator is n=len(returns), if True the denominator is n-k. Default is False.
        adjust_daily (bool, optional): If True, the autocovariances used in estimating the
            autocorrelations are computed by multiplying returns on same days only. This means that
            the larger the lag, the less data we use to compute the autocorrelations. Default is False.
        transformation (func, optional): A function to be applied to the returns, e.g. the absolute value function.
            Default is the identity function.
        upload_path (string or None, optional): The path where to save the figure. If not None, the figure
            is saved according to the input path. If None, the figure is not saved. Default is None.

        Returns
        -------
        None.
        """
        df = self.df.copy()
        start_date = datetime.datetime(year=start_date.year, month=start_date.month, day=start_date.day, hour=0, minute=0)
        end_date = datetime.datetime(year=end_date.year, month=end_date.month, day=end_date.day, hour=23, minute=59)
        df = df.loc[(df['Time Datetime'] >= start_date) & (df['Time Datetime'] <= end_date)].copy()
        df.reset_index(drop=True, inplace=True)
        start_date = df['Time Datetime'].iloc[0]
        end_date = df['Time Datetime'].iloc[-1]
        start_date = datetime.datetime(year=start_date.year, month=start_date.month, day=start_date.day, hour=0, minute=0)
        end_date = datetime.datetime(year=end_date.year, month=end_date.month, day=end_date.day, hour=23, minute=59)

        # Filter out the returns at the beginning and end of each day
        daily_start_date = datetime.datetime(year=1900, month=1, day=1, hour=9, minute=30)
        daily_end_date = datetime.datetime(year=1900, month=1, day=1, hour=16, minute=0)
        daily_start_date = daily_start_date + dateutil.relativedelta.relativedelta(minutes=filter_minutes)
        daily_end_date = daily_end_date + dateutil.relativedelta.relativedelta(minutes=-filter_minutes)
        df = df.loc[(df['Hour Datetime'] >= daily_start_date) & (df['Hour Datetime'] <= daily_end_date)].copy()

        df.set_index('Time', inplace=True)
        returns = df['Log-returns 1-min Period']
        returns.name = 'Log-returns'
        returns = transformation(returns)

        acf, pacf, confint = self.calculate_autocorrelations(returns, nlags=nlags, alpha=alpha,
            adjust_denominator=adjust_denominator, adjust_daily=adjust_daily)

        self.plot_acf_pacf(acf, pacf, confint, start_date, end_date, alpha=alpha, upload_path=upload_path)




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
        df.rename(columns={'index': 'Index'}, inplace=True)
        list_indices = df.groupby('Date').first()['Index'].to_list()
        df.drop(index=list_indices, inplace=True)
        df.drop('Index', axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        if filter_minutes != 0:
            # Filter out the returns at the beginning and end of each day
            daily_start_date = datetime.datetime(year=1900, month=1, day=1, hour=9, minute=30)
            daily_end_date = datetime.datetime(year=1900, month=1, day=1, hour=16, minute=0)
            daily_start_date = daily_start_date + dateutil.relativedelta.relativedelta(minutes=filter_minutes + period)
            daily_end_date = daily_end_date + dateutil.relativedelta.relativedelta(minutes=-filter_minutes)
            df = df.loc[(df['Hour Datetime'] >= daily_start_date) & (df['Hour Datetime'] <= daily_end_date)].copy()

        if normalize:
            dict_daily_averages = df.groupby('Date').mean().to_dict()['Log-returns']
            dict_daily_sd = df.groupby('Date').std().to_dict()['Log-returns']
            return np.array(df.apply(lambda row: (row['Log-returns'] - dict_daily_averages[row['Date']]) / dict_daily_sd[row['Date']], axis=1).to_list())
        else:
            return np.array(df['Log-returns'].to_list())


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
        denominator = std_returns ** moment
        empirical_moment = numerator / denominator
        
        # Calculate the theoretical moment of the standard normal distribution
        if moment % 2:
            moment_snd = 0
        else:
            moment_snd = np.math.factorial(moment) / ((2 ** (moment / 2)) * np.math.factorial(int(moment / 2)))

        excess_moment = empirical_moment - moment_snd
        return excess_moment


    def compute_moment_between_two_dates(self, df, start_date, end_date, moment=4, filter_minutes=15, normalize=False):
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
        df_log_returns = df.loc[(df['Time Datetime'] > start_date) & (df['Time Datetime'] <= end_date)].copy()
        df_log_returns.reset_index(drop=True, inplace=True)
        list_periods = list(range(1, 61))
        list_excess_moments = []
        for period in list_periods:
            column_name = 'Log-returns {}-min Period'.format(period)
            df_temp = df_log_returns.loc[df_log_returns[column_name].notnull()][['Date', 'Hour Datetime', column_name]].copy()
            df_temp.reset_index(drop=True, inplace=True)
            df_temp.rename(columns={column_name: 'Log-returns'}, inplace=True)
            returns = self.compute_log_returns(df_temp, period, filter_minutes=filter_minutes, normalize=normalize)
            excess_moment = self.calculate_excess_moment(returns, moment=moment)
            list_excess_moments.append(excess_moment)
        
        return np.array(list_excess_moments).reshape((1, 60))


    @staticmethod
    def plot_moments(arr_moment_average_standard, arr_moment_one_period_standard, arr_moment_average_normalized,
        arr_moment_one_period_normalized, start_date, end_date, frequency, moment=4, upload_path=None):
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
        dict_moment_names = {1: 'mean', 2: 'variance', 3: 'skewness', 4: 'excess kurtosis'}
        title_moment = string.capwords(dict_moment_names[moment])
        legend_moment = dict_moment_names[moment].capitalize()

        nb_days_per_period = str(np.round(frequency / 391, decimals=0)).split('.0')[0]
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        
        axs[0].plot(list(range(1, 61)), arr_moment_average_standard, label='{} computed as an average of {}-day subperiods'.format(legend_moment, nb_days_per_period))
        axs[0].plot(list(range(1, 61)), arr_moment_one_period_standard, label='{} computed over the whole period'.format(legend_moment))
        axs[0].set_xlabel('Period (in minutes)')
        axs[0].set_ylabel(legend_moment)
        axs[0].axhline(y=0, color='red', linestyle='--')
        axs[0].set_title("{} of Standard Returns".format(title_moment))
        
        axs[1].plot(list(range(1, 61)), arr_moment_average_normalized)
        axs[1].plot(list(range(1, 61)), arr_moment_one_period_normalized)
        axs[1].set_xlabel('Period (in minutes)')
        axs[1].set_ylabel(legend_moment)
        axs[1].axhline(y=0, color='red', linestyle='--')
        axs[1].set_title("{} of Daily Normalized Returns".format(title_moment))
        
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.674, 0.87))
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        fig.suptitle("{} as a function of timescale from {} to {}".format(legend_moment, start_date, end_date), fontsize=22, y=0.943)
        fig.subplots_adjust(top=0.68)

        if upload_path is not None:
            plt.savefig(upload_path)
        plt.show()


    def calculate_and_plot_moments(self, start_date, end_date, frequency, moment=4, filter_minutes=15, upload_path=None):
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
        start_date = datetime.datetime(year=start_date.year, month=start_date.month, day=start_date.day, hour=0, minute=0)
        end_date = datetime.datetime(year=end_date.year, month=end_date.month, day=end_date.day, hour=23, minute=59)
        df = df.loc[(df['Time Datetime'] >= start_date) & (df['Time Datetime'] <= end_date)].copy()
        df.reset_index(drop=True, inplace=True)
        start_date = df['Time Datetime'].iloc[0]
        end_date = df['Time Datetime'].iloc[-1]
        start_date = datetime.datetime(year=start_date.year, month=start_date.month, day=start_date.day, hour=0, minute=0)
        end_date = datetime.datetime(year=end_date.year, month=end_date.month, day=end_date.day, hour=23, minute=59)

        list_dates = []
        i = df.loc[df['Time Datetime'] <= end_date].index[-1]
        current_date = end_date
        while current_date >= start_date:
            list_dates.append(current_date)
            i -= frequency
            if i < 0:
                break
            current_date = df.iloc[i]['Time Datetime']
            current_date = df.loc[df['Date'] == current_date.strftime('%Y-%m-%d')].iloc[0]['Time Datetime']
            i = df.loc[df['Date'] == current_date.strftime('%Y-%m-%d')].iloc[0].name
        
        arr_moment_full_standard = np.empty((0, 60))
        arr_moment_full_normalized = np.empty((0, 60))
        for i in range(len(list_dates) - 1):
            start_date_temp = list_dates[i+1]
            end_date_temp = list_dates[i]
            arr_moment_standard = self.compute_moment_between_two_dates(df, start_date_temp, end_date_temp, moment=moment, filter_minutes=filter_minutes, normalize=False)
            arr_moment_full_standard = np.concatenate([arr_moment_full_standard, arr_moment_standard], axis=0)
            arr_moment_normalized = self.compute_moment_between_two_dates(df, start_date_temp, end_date_temp, moment=moment, filter_minutes=filter_minutes, normalize=True)
            arr_moment_full_normalized = np.concatenate([arr_moment_full_normalized, arr_moment_normalized], axis=0)
        
        arr_moment_average_standard = np.mean(arr_moment_full_standard, axis=0)
        arr_moment_average_normalized = np.mean(arr_moment_full_normalized, axis=0)
        arr_moment_one_period_standard = self.compute_moment_between_two_dates(df, start_date, end_date, moment=moment, filter_minutes=filter_minutes, normalize=False).reshape(60)
        arr_moment_one_period_normalized = self.compute_moment_between_two_dates(df, start_date, end_date, moment=moment, filter_minutes=filter_minutes, normalize=True).reshape(60)

        self.plot_moments(arr_moment_average_standard, arr_moment_one_period_standard, arr_moment_average_normalized,
            arr_moment_one_period_normalized, start_date, end_date, frequency, moment=moment, upload_path=upload_path)


