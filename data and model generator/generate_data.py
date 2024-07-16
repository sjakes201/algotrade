import os
import pandas as pd
import numpy as np
import pytz
from calc_indicators import calculate_vwap_1800_to_1800, calculate_adx, calculate_momentum, calculate_wilders_rsi, calculate_sma, calculate_ema, calculate_stochastic, calculate_macd, calculate_bollinger_bands, calculate_pivot_points, calculate_historical_support_resistance
from calc_features import calc_volume_ratio, market_regime, calculate_volatility
from indiv_trading_strategies import momentum_trader, mean_reversion_trader, breakout_trader, reversal_trader, vwap_trend_trader, ema_trader_12_26, scalping_trader, heikin_ashi_trader
from simulate_trade import simulate_trade, simulate_trade_threaded
import time

# [--] HELPER FUNCTIONS [--]

# timeframe / bounding functions

def bound_times(df):     
    # 9:30 AM - 11:30 AM ET is morning, 11:30 - 14:00 is midday, 14:00 - 16:00 is closing
    # start_time = pd.to_datetime('9:30:00').time()
    # end_time = pd.to_datetime('11:30:00').time()
    
    # start_time = pd.to_datetime('11:30:00').time()
    # end_time = pd.to_datetime('14:00:00').time()
    
    start_time = pd.to_datetime('14:00:00').time()
    end_time = pd.to_datetime('16:00:00').time()
    
    df = df[(df['time_est'] >= start_time) & (df['time_est'] <= end_time) & (df['day_of_week_est'] < 5)]    
    return df
    
def bound_expirations(df):
    """
        From 2023-05-05 to 2023-06-15 it should only be ESM3
        From 2023-06-17 to 2023-09-14 it should be ESU3
        From 2023-09-16 to 2023-12-14 it should be ESZ3
        From 2023-12-16 to 2024-03-14 it should be ESH4 
        From 2024-03-15 to 2024-06-17 it should be ESM4
        From 2024-06-18 to 2024-09-20 it should be ESU4
    """
    est = pytz.timezone('US/Eastern')
    periods = [
    {"next_symbol": "ESM3", "start_date": "2023-05-05", "end_date": "2023-06-15"},
    {"next_symbol": "ESU3", "start_date": "2023-06-16", "end_date": "2023-09-14"},
    {"next_symbol": "ESZ3", "start_date": "2023-09-15", "end_date": "2023-12-14"},
    {"next_symbol": "ESH4", "start_date": "2023-12-15", "end_date": "2024-03-14"},
    {"next_symbol": "ESM4", "start_date": "2024-03-15", "end_date": "2024-06-20"},
    {"next_symbol": "ESU4", "start_date": "2024-06-21", "end_date": "2024-09-19"}
    ]

    
    filtered_data = pd.DataFrame()
    for period in periods:
        start_date = pd.to_datetime(period["start_date"]).tz_localize(est).tz_convert('UTC')
        end_date = pd.to_datetime(period["end_date"]).tz_localize(est).tz_convert('UTC')
        next_symbol = period["next_symbol"]
        
        mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date) & (df['symbol'] == next_symbol)
        expiry_data = df[mask]
        
        filtered_data = pd.concat([filtered_data, expiry_data])
    
    return filtered_data

# general helpers

def add_datetime(df):
    # Convert UTC timestamp to datetime
    df['datetime'] = pd.to_datetime(df['ts_event'] / 1000000, unit='ms', utc=True)
    df = df.sort_values('datetime')
    
    # Define the EST timezone
    est = pytz.timezone('US/Eastern')
    
    # Convert UTC datetime to EST
    df['datetime_est'] = df['datetime'].dt.tz_convert(est)

    # Extract time and day of the week in EST
    df['time_est'] = df['datetime_est'].dt.time
    df['day_of_week_est'] = df['datetime_est'].dt.dayofweek
    
    return df

# [--] MAIN FUNCTIONS [--]

def process_data(input_folder, output_file):

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv') and 'ohlcv' in file_name:
            file_path = os.path.join(input_folder, file_name)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # pre indicator calculations
            df = add_datetime(df)

            # instrument filtering
            df = bound_expirations(df)
            df = df.drop_duplicates(subset='datetime_est')

            tasks = {
                "momentum_strategy": True,
                "mean_reversion_strategy": True,
                "breakout_strategy": True,
                "reversal_strategy": True,
                "vwap_trend_strategy": True,
                "ema_crossover_strategy": True,
                "stochastic_strategy": True,
                "scalping_strategy": True,
                "heikin_ashi_strategy": True,
                "meta_model_features": True,
            }

            # INDICATORS USED BY MULTIPLE STRATEGIES:
            if tasks["momentum_strategy"] or tasks["mean_reversion_strategy"] or tasks["breakout_strategy"] or tasks["reversal_strategy"] or tasks["vwap_trend_strategy"] or tasks["ema_crossover_strategy"] or tasks["scalping_strategy"] or tasks["heikin_ashi_strategy"] or tasks["meta_model_features"]:
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calc_volume_ratio(x))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_wilders_rsi(x))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_ema(x, 50))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_macd(x))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_vwap_1800_to_1800(x))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_sma(x, 50))



            # MOMENTUM STRATEGY INDICATORS (rsi, sma, macd, momentum)
            if tasks['momentum_strategy']:
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_momentum(x, 12))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: momentum_trader(x))
            
            # MEAN REVERSION STRATEGY INDICATORS (bollinger upper & lower, rsi)
            if tasks['mean_reversion_strategy']:
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_bollinger_bands(x))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: mean_reversion_trader(x))
            
            # BREAKOUT TRADING STRATEGY INDICATORS ()
            if tasks["breakout_strategy"]:
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_pivot_points(x))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_historical_support_resistance(x))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: breakout_trader(x))
                
            # REVERSAL TRADING STRATEGY
            if tasks["reversal_strategy"]:
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_adx(x, 14))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: reversal_trader(x))
            
            # VWAP TREND STRATEGY
            if tasks["vwap_trend_strategy"]:
                df = df.sort_values(by='datetime_est') # VWAP calculation messes it up need to resort
                # df = df.drop("vwap_session")
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_ema(x, 5))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: vwap_trend_trader(x))

            # 12 26 EMA CROSSOVER STRATEGY
            if tasks["ema_crossover_strategy"]:
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_ema(x, 12))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_ema(x, 26))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: ema_trader_12_26(x))
                df = df.sort_values(by='datetime_est')
                
            # STOICHASTIC STRATEGY ------- this isn't used and has no trading strategy to accompany it?
            if tasks["stochastic_strategy"] and False: 
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_stochastic(x))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_ema(x, 200))
   
            # SCALPING STRATEGY
            if tasks["scalping_strategy"]:
                df = df.groupby('symbol', group_keys=False).apply(lambda x: scalping_trader(x))
            
            # HEIKIN ASHI STRATEGY
            if tasks["heikin_ashi_strategy"]:
                df = df.groupby('symbol', group_keys=False).apply(lambda x: heikin_ashi_trader(x))
                
            # META-MODEL FEATURES other than trading strategy outcomes 
            if tasks["meta_model_features"]:
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_sma(x, 200))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_sma(x, 100))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_sma(x, 20))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_adx(x, 50))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: market_regime(x))
                df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_volatility(x, 45))
            
            
            # print('Starting profit calculations')
            # df = df.groupby('symbol', group_keys=False).apply(lambda x: simulate_trade(x))
            # df = simulate_trade_threaded(df)
            # print('Profit calculations complete')
            
            # bound to target trading hours and do NTCAE calculations
            df = bound_times(df)
            

            # clean up
            df['datetime_est'] = df['datetime_est'].astype(str).str[:-6]
            to_keep = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'datetime_est', 'momentum_signal',
                       'mean_reversion_signal', 'breakout_signal', 'vwap_trend_signal', 'reversal_signal', 'ema_12_26_signal',
                       'scalping_signal', 'heikin_ashi_signal', 'volatility', 'market_regime_ADX', 'market_regime_MA', 
                       'remaining_in_trend', 'NTCAE', 'buy_profit', 'sell_profit', 'total_buys', 'total_sells']
            to_delete = df.columns.difference(to_keep)
            df = df.drop(columns=to_delete)
    
            df.to_csv(output_file, index=False)


if __name__ == '__main__':
    input_folder = './futures_data/most_recent_2'  
    output_file =  './futures_data/most_recent_2/closing.csv'
    start = time.time()
    process_data(input_folder, output_file)

    end = time.time()
    print(end-start)
