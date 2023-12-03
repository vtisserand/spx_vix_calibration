import pandas as pd
from plotting_tools import *


def main():
    wd = "E:\\OneDrive\\Documents\\MIASHS 2022-10\\ENSAE_2023-2024\\Calibration VIX-SPX\\Data\\Clean_data\\"
    df = pd.read_parquet(wd + "spx500_intraday_1min_barchart_log_returns_short.parquet")
    dataframe_tools = DataFrameTools()
    df = dataframe_tools.construct_datetime_columns(df, construct_time=True, construct_date=True, construct_hour=True)

    start_date = datetime.datetime(year=2023, month=8, day=1)
    end_date = datetime.datetime(year=2023, month=11, day=30)

    filter_minutes = 30
    nlags = 20
    alpha = 0.05
    title = "Autocorrelations"
    autocorrelation_tools = AutocorrelationTools(df)

    autocorrelation_tools.plot_autocorrelations(start_date, end_date, filter_minutes=filter_minutes, nlags=nlags, alpha=alpha,
        adjust_denominator=False, adjust_daily=False, transformation=lambda x: x, title=title, upload_path=None)
    autocorrelation_tools.plot_autocorrelations(start_date, end_date, filter_minutes=filter_minutes, nlags=nlags, alpha=alpha,
        adjust_denominator=False, adjust_daily=False, transformation=lambda x: np.abs(x), title=title, upload_path=None)

    
    start_date = datetime.datetime(year=2023, month=1, day=4)
    end_date = datetime.datetime(year=2023, month=11, day=4)
    frequency = 12242
    kurtosis_tools = KurtosisTools(df)
    kurtosis_tools.calculate_and_plot_kurtosis(start_date, end_date, frequency)
    

if __name__ == '__main__':
    main()
