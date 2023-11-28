import pandas as pd
import datetime
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import statsmodels.api as sm


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
    def plot_autocorrelations_without_pacf(arr_returns, nlags, confidence_level, title):
        arr_acf = sm.tsa.stattools.acf(arr_returns, nlags=nlags, alpha=None)
        arr_acf = arr_acf[1:]
        x = np.array(range(1, nlags+1))

        fig, ax = plt.subplots(figsize=(7, 4))
        color = '#1f77b4'
        margin = 0.01
        max_abs_value = max(abs(min(arr_acf)), abs(max(arr_acf)))
        max_y_value = max_abs_value + margin
        min_y_value = -max_abs_value - margin
        confidence_band = scipy.stats.norm.ppf(1-((1-confidence_level)/2)) / np.sqrt(len(arr_returns))
        str_confidence_level = str(np.round(100*confidence_level, decimals=0)).split('.0')[0]

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
    
    @staticmethod
    def plot_autocorrelations_with_pacf(arr_returns, nlags, confidence_level, title):
        arr_acf = sm.tsa.stattools.acf(arr_returns, nlags=nlags, alpha=None)
        arr_pacf = sm.tsa.stattools.pacf(arr_returns, nlags=nlags, method='ywadjusted', alpha=None)
        arr_acf = arr_acf[1:]
        arr_pacf = arr_pacf[1:]
        x = np.array(range(1, nlags+1))

        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(7, 4))
        color = '#1f77b4'
        margin = 0.01
        max_abs_value = max(abs(min(min(arr_acf), min(arr_pacf))), abs(max(max(arr_acf), max(arr_pacf))))
        max_y_value = max_abs_value + margin
        min_y_value = -max_abs_value - margin
        confidence_band = scipy.stats.norm.ppf(1-((1-confidence_level)/2)) / np.sqrt(len(arr_returns))
        str_confidence_level = str(np.round(100*confidence_level, decimals=0)).split('.0')[0]

        # Plot ACF using stem plot
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        markerline, stemlines, baseline = ax1.stem(x, arr_acf, linefmt='-', basefmt=' ')
        plt.setp(stemlines, 'color', color) # Set stem line color
        plt.setp(markerline, 'color', color) # Set marker line color
        ax1.set_ylabel('ACF')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_ylim(min_y_value, max_y_value)
        odd_x_ticks = np.arange(1, nlags + 1, 2)
        ax1.set_xticks(odd_x_ticks)
        ax1.set_xticklabels(odd_x_ticks)
        ax1.axhline(y=confidence_band, color='red', linestyle='--', label="{}% confidence interval".format(str_confidence_level))
        ax1.axhline(y=-confidence_band, color='red', linestyle='--')

        # Plot PACF using stem plot
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        markerline, stemlines, baseline = ax2.stem(x, arr_pacf, linefmt='-', basefmt=' ')
        plt.setp(stemlines, 'color', color) # Set stem line color
        plt.setp(markerline, 'color', color) # Set marker line color
        ax2.set_xlabel('Lag (in minutes)')
        ax2.set_ylabel('PACF')
        ax2.set_ylim(min_y_value, max_y_value)
        ax2.axhline(y=confidence_band, color='red', linestyle='--')
        ax2.axhline(y=-confidence_band, color='red', linestyle='--')

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.9, 0.985))
        fig.suptitle(title, fontsize=22, x=0.35, y=0.97)
        fig.subplots_adjust(hspace=0.2, top=0.85)
        plt.show()

    def plot_autocorrelations(self, start_date, end_date, with_pacf, nlags=20, confidence_level=0.95, title="Autocorrelations"):
        self.df = self.df.loc[(self.df['Time Datetime'] >= start_date) & (self.df['Time Datetime'] <= end_date)].copy()
        self.df.reset_index(inplace=True)
        daily_start_date = datetime.datetime(year=1900, month=1, day=1, hour=10, minute=0)
        daily_end_date = datetime.datetime(year=1900, month=1, day=1, hour=15, minute=30)
        self.df = self.df.loc[(self.df['Hour Datetime'] >= daily_start_date) & (self.df['Hour Datetime'] <= daily_end_date)].copy()
        arr_returns = np.diff(np.log(self.df['Last']))
        if with_pacf:
            self.plot_autocorrelations_with_pacf(arr_returns, nlags, confidence_level, title)
        else:
            self.plot_autocorrelations_without_pacf(arr_returns, nlags, confidence_level, title)


class KurtosisTools:

    def __init__(self, df):
        self.df = df.copy()

    @staticmethod
    def compute_log_returns(df, period, normalize):
        # Removing the first log-return of each day (since it was calculated using the price of the previous day)
        column_name = 'Log-returns {}-min Period'.format(period)
        df_one_period = df.loc[df[column_name].notnull()][['Time', 'Date', column_name]].copy()
        df_one_period.reset_index(drop=True, inplace=True)
        df_one_period.reset_index(inplace=True)
        df_one_period.rename(columns={'index': 'Index'}, inplace=True)
        list_indices = df_one_period[['Index', 'Time', 'Date', column_name]].groupby('Date').first()['Index'].to_list()
        df_one_period.drop(index=list_indices, inplace=True)
        df_one_period.drop(['Index', 'Time'], axis=1, inplace=True)
        df_one_period.reset_index(drop=True, inplace=True)
        
        if normalize:
            dict_daily_averages = df_one_period.groupby('Date').mean().to_dict()[column_name]
            dict_daily_sd = df_one_period.groupby('Date').std().to_dict()[column_name]
            return np.array(df_one_period.apply(lambda row: (row[column_name] - dict_daily_averages[row['Date']]) / dict_daily_sd[row['Date']], axis=1).to_list())
        else:
            return np.array(df_one_period[column_name].to_list())

    @staticmethod
    def calculate_excess_kurtosis(arr_returns):
        mean_returns = np.mean(arr_returns)
        std_returns = np.std(arr_returns)
        
        # Calculate kurtosis
        numerator = np.mean((arr_returns - mean_returns) ** 4)
        denominator = std_returns ** 4
        
        kurtosis = numerator / denominator - 3
        return kurtosis

    def compute_kurtosis_between_two_dates(self, start_date, end_date, normalize):
        df_log_returns = self.df.loc[(self.df['Time Datetime'] > start_date) & (self.df['Time Datetime'] <= end_date)].copy()
        df_log_returns.reset_index(drop=True, inplace=True)
        
        list_periods = list(range(1, 61))
        list_excess_kurtosis = []
        for period in list_periods:
            arr_returns = self.compute_log_returns(df_log_returns, period, normalize)
            kurtosis = self.calculate_excess_kurtosis(arr_returns)
            list_excess_kurtosis.append(kurtosis)
        
        return np.array(list_excess_kurtosis).reshape((1, 60))


    def compute_kurtosis_to_plot(self, start_date, end_date, frequency, normalize):
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
        arr_kurtosis_one_period = self.compute_kurtosis_between_two_dates(start_date, end_date, normalize)
        return arr_kurtosis_average, arr_kurtosis_one_period.reshape(60)

    @staticmethod
    def plot_kurtosis(arr_kurtosis_average_standard,
                      arr_kurtosis_one_period_standard,
                      arr_kurtosis_average_normalized,
                      arr_kurtosis_one_period_normalized,
                      start_date,
                      end_date,
                      frequency):
        nb_days_per_period = str(np.round(frequency / 391, decimals=0)).split('.0')[0]
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        
        axs[0].plot(list(range(1, 61)), arr_kurtosis_average_standard, label='Kurtosis computed as an average of {}-day subperiods'.format(nb_days_per_period))
        axs[0].plot(list(range(1, 61)), arr_kurtosis_one_period_standard, label='Kurtosis computed over the whole period')
        axs[0].set_xlabel('Period (in minutes)')
        axs[0].set_ylabel('Excess kurtosis')
        axs[0].axhline(y=3, color='red', linestyle='--')
        axs[0].set_title("Excess Kurtosis of Standard Returns")
        
        axs[1].plot(list(range(1, 61)), arr_kurtosis_average_normalized)
        axs[1].plot(list(range(1, 61)), arr_kurtosis_one_period_normalized)
        axs[1].set_xlabel('Period (in minutes)')
        axs[1].set_ylabel('Excess kurtosis')
        axs[1].axhline(y=3, color='red', linestyle='--')
        axs[1].set_title("Excess Kurtosis of Daily Normalized Returns")
        
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.648, 0.87))
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        fig.suptitle("Kurtosis as a function of timescale from {} to {}".format(start_date, end_date), fontsize=22, y=0.943)
        fig.subplots_adjust(top=0.68)
        plt.show()

    def calculate_and_plot_kurtosis(self, start_date, end_date, frequency):
        arr_kurtosis_average_standard, arr_kurtosis_one_period_standard = self.compute_kurtosis_to_plot(start_date, end_date, frequency, False)
        arr_kurtosis_average_normalized, arr_kurtosis_one_period_normalized = self.compute_kurtosis_to_plot(start_date, end_date, frequency, True)
        self.plot_kurtosis(arr_kurtosis_average_standard,
                           arr_kurtosis_one_period_standard,
                           arr_kurtosis_average_normalized,
                           arr_kurtosis_one_period_normalized,
                           start_date,
                           end_date,
                           frequency)






