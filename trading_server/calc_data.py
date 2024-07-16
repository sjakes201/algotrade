from indicator_funcs import accumulating_features, calc_volume_ratio, add_total_signals, market_regime, calculate_volatility, calculate_adx, calculate_historical_support_resistance, calculate_pivot_points, calculate_vwap_1800_to_1800, calculate_wilders_rsi, calculate_ema, calculate_sma, calculate_macd, calculate_momentum, calculate_bollinger_bands
from indiv_trading_strategies import vwap_trend_trader, momentum_trader, mean_reversion_trader, breakout_trader, reversal_trader, vwap_trend_trader, ema_trader_12_26, scalping_trader, heikin_ashi_trader
import pandas as pd
import numpy as np
import pytz
from pytz import timezone
import time

def calc_features(df):
    try:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                
        df = heikin_ashi_trader(df)
        df = calculate_vwap_1800_to_1800(df)
        indicators_and_periods = [
            (lambda x: calculate_sma(x, 50), 50),
            (calc_volume_ratio, 14),
            (calculate_wilders_rsi, 14),
            (lambda x: calculate_ema(x, 50), 50),
            (calculate_macd, 26),
            (lambda x: calculate_momentum(x, 12), 13),
            (momentum_trader, 1),
            (calculate_bollinger_bands, 20),
            (mean_reversion_trader, 1),
            (calculate_pivot_points, 2),
            (calculate_historical_support_resistance, 21),
            (breakout_trader, 1),
            (calculate_adx, 14),
            (reversal_trader, 3),
            (lambda x: calculate_ema(x, 5), 5),
            (lambda x: calculate_ema(x, 12), 12),
            (lambda x: calculate_ema(x, 26), 26),
            (ema_trader_12_26, 1),
            (scalping_trader, 2),
            (lambda x: calculate_sma(x, 200), 200),
            (lambda x: calculate_sma(x, 100), 100),
            (lambda x: calculate_sma(x, 20), 20),
            (lambda x: calculate_adx(x, 50), 50),
            (market_regime, 1),
            (lambda x: calculate_volatility(x, 45), 45),
            (vwap_trend_trader, 3),
            (accumulating_features, 5),
            (add_total_signals, 1),
        ]
        
        for func, period in indicators_and_periods:
            if len(df) >= period:
                # Apply the function to the last X rows of the DataFrame
                df_last_period = df.iloc[-period:].copy()
                df_last_period = func(df_last_period)
                
                # Merge the result back into the original DataFrame
                for column in df_last_period.columns:
                    if column not in df.columns:
                        # Initialize the column with the appropriate dtype
                        example_value = df_last_period[column].iloc[-1]
                        if pd.api.types.is_numeric_dtype(example_value):
                            df[column] = np.nan
                        else:
                            df[column] = None  # Use None for non-numeric types

                    # Ensure matching lengths
                    if len(df_last_period[column]) == len(df.iloc[-period:, df.columns.get_loc(column)]):
                        # Convert the data types if necessary
                        if df[column].dtype != df_last_period[column].dtype:
                            df[column] = df[column].astype(object)  # Cast to object to accommodate mixed types
                        df.iloc[-period:, df.columns.get_loc(column)] = df_last_period[column].values
                    else:
                        print(f"Length mismatch for column {column}: {len(df_last_period[column])} != {len(df.iloc[-period:, df.columns.get_loc(column)])}")
                        raise ValueError("Length mismatch between keys and values")
        
        return df
    except Exception as e:
        print(f"An error occurred in calc_features: {e}")
        return df  # Return the original DataFrame in case of error
    

def main():
    df = pd.read_csv('test_data_1m.csv')
    df['datetime_utc'] = pd.to_datetime(df['ts_event'] / 1000000, unit='ms', utc=True)
    # Convert the UTC datetime to EST
    eastern = timezone('US/Eastern')

    start = time.time()
    df = calc_features(df)
    end = time.time()
    print(f"Calculation took {end - start} seconds")

    df.to_csv('RESULTS_test_data_1m.csv', index=False)