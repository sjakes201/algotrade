import pandas as pd
import numpy as np
from calc_features import add_total_signals
from simulate_trade import simulate_trade_threaded

def determine_label(df, profit_minimum = 4):
    
    labels = np.full(len(df), "HOLD", dtype=object)
    weights = np.full(len(df), 1, dtype=float)
    
    # Count how many non-HOLD signals there are per row, excluding perpetual signals like heikin ashi
    signal_counts = (df[['momentum_signal', 'mean_reversion_signal', 'breakout_signal', 'reversal_signal', 'vwap_trend_signal', 'ema_12_26_signal', 'scalping_signal']] != 'HOLD').sum(axis=1)
    # Vectorized conditions for buy, sell, and hold
    buy_condition = (df['buy_profit'] >= df['sell_profit']) & (df['buy_profit'] >= profit_minimum)
    sell_condition = (df['sell_profit'] > df['buy_profit']) & (df['sell_profit'] >= profit_minimum)
    
   
    labels[buy_condition] = "BUY"
    labels[sell_condition] = "SELL"
    
    weights[buy_condition] = df.loc[buy_condition, 'buy_profit'].abs()
    weights[sell_condition] = df.loc[sell_condition, 'sell_profit'].abs()
    
    
    df['TRADE_SIGNAL'] = labels
    df['TRADE_WEIGHT'] = weights
    
    for i in range(len(df)):
        if labels[i] == "HOLD":
            continue
        if signal_counts[i] == 0:
            weights[i] = 1
        elif signal_counts[i] == 1:
            weights[i] *= 0.5
        elif signal_counts[i] >= 3:
            weights[i] *= 1.5
    
    df['SUPPRESSED_WEIGHTS'] = weights
    
    return df
    
def data_quality(df):
    df['temp_date'] = pd.to_datetime(df['datetime_est']).dt.date
    minute_counts = df['temp_date'].value_counts()
    print(minute_counts)
    # Morning should have 121 minutes, midday 151, closing 121
    valid_dates = minute_counts[minute_counts == 121].index
    df = df[df['temp_date'].isin(valid_dates)]
    df = df.drop(columns=['temp_date'])
    return df

def accumulating_features(df):
    accumulating_columns = ['momentum_signal', 'mean_reversion_signal', 'breakout_signal', 'reversal_signal', 'vwap_trend_signal', 'ema_12_26_signal', 'scalping_signal', 'heikin_ashi_signal']
    
    for col in accumulating_columns:
        buy_accum = np.full(len(df), 0)
        sell_accum = np.full(len(df), 0)
        
        buy_accum[4:] = (df[col] == "BUY").rolling(window=5).sum().values[4:]
        sell_accum[4:] = (df[col] == "SELL").rolling(window=5).sum().values[4:]
        
        df[f"{col}_buy_accum"] = buy_accum
        df[f"{col}_sell_accum"] = sell_accum
    return df



if __name__ == '__main__':
    input_file = './futures_data/most_recent_2/closing.csv'  
    output_file =  './futures_data/most_recent_2/closing_testing.csv'  
    df = pd.read_csv(input_file)
    # df = simulate_trade_threaded(df)
    df[['open', 'high','low','close']] = df[['open', 'high','low','close']] / 1e9
    # df = determine_label(df)
    df = data_quality(df)
    df = add_total_signals(df)
    df = df.groupby('symbol', group_keys=False).apply(lambda x: accumulating_features(x))

    
    df.to_csv(output_file, index=False)
