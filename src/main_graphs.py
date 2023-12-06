import pandas as pd
import numpy as np
from plotting_tools import *


def main():
    wd = "E:\\OneDrive\\Documents\\MIASHS 2022-10\\ENSAE_2023-2024\\Calibration VIX-SPX\\Data\\Clean_data\\"
    df = pd.read_parquet(wd + "spx500_intraday_1min_barchart_log_returns.parquet")
    dataframe_tools = DataFrameTools()
    df = dataframe_tools.construct_datetime_columns(df, construct_time=True, construct_date=True, construct_hour=True)

    start_date = datetime.datetime(year=2023, month=8, day=1)
    end_date = datetime.datetime(year=2023, month=11, day=30)



    filter_minutes = 15
    nlags = 20
    alpha = 0.05
    autocorrelation_tools = CrossCorrelationTools(df)
    
    upload_path = "E:\\OneDrive\\Documents\\MIASHS 2022-10\\ENSAE_2023-2024\\Calibration VIX-SPX\\Rapport\\Stylized facts\\Linear_Autocorrelations_standard_returns_adjust_daily_False_filter_15_20_lags.pdf"
    autocorrelation_tools.plot_autocorrelations(start_date, end_date, filter_minutes=filter_minutes, nlags=nlags, alpha=alpha,
        adjust_denominator=False, adjust_daily=False, transformation=lambda x: x, negative_lags=False, upload_path=upload_path)
    
    
    start_date = datetime.datetime(year=2023, month=1, day=1)
    end_date = datetime.datetime(year=2023, month=11, day=30)
    frequency = 12242
    filter_minutes = 15
    moment = 3
    kurtosis_tools = MomentTools(df)
    upload_path = "E:\\OneDrive\\Documents\\MIASHS 2022-10\\ENSAE_2023-2024\\Calibration VIX-SPX\\Rapport\\Stylized facts\\Skewness_frequency_12242_filter_15.pdf"
    kurtosis_tools.calculate_and_plot_moments(start_date, end_date, frequency, moment=moment, filter_minutes=filter_minutes, upload_path=upload_path)
    
    
    
    

if __name__ == '__main__':
    main()
