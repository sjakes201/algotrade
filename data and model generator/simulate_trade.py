import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def simulate_trade(df):
    max_look_ahead = 60  # Number of minutes
    buy_profits = np.full(len(df), np.nan)
    sell_profits = np.full(len(df), np.nan)
    
    for i in range(len(df)):
        entry_price = df['close'].iloc[i]
        
        # Trailing stop for buy profit
        trailing_stop_threshold = 5
        stop_loss_price = entry_price - trailing_stop_threshold
        
        for j in range(1, max_look_ahead):
            if i + j >= len(df):
                buy_profits[i] = df['close'].iloc[-1] - entry_price
                break
                
            new_price = df['close'].iloc[i + j]
            if new_price > stop_loss_price + trailing_stop_threshold:
                stop_loss_price = new_price - trailing_stop_threshold
                
                unrealized_profit = new_price - entry_price
                if unrealized_profit >= 10:
                    trailing_stop_threshold = 1
                elif unrealized_profit >= 8:
                    trailing_stop_threshold = 2
                elif unrealized_profit >= 5:
                    trailing_stop_threshold = 3
                elif unrealized_profit >= 47:
                    trailing_stop_threshold = 4
                
            if new_price <= stop_loss_price:
                buy_profits[i] = stop_loss_price - entry_price
                break
                
            if j == max_look_ahead - 1:
                buy_profits[i] = stop_loss_price - entry_price
        
        # Trailing stop for sell profit
        trailing_stop_threshold = 5
        stop_loss_price = entry_price + trailing_stop_threshold
        
        for j in range(1, max_look_ahead):
            if i + j >= len(df):
                sell_profits[i] = entry_price - df['close'].iloc[-1]
                break
                
            new_price = df['close'].iloc[i + j]
            if new_price < stop_loss_price - trailing_stop_threshold:
                stop_loss_price = new_price + trailing_stop_threshold
                
                unrealized_profit = entry_price - new_price
                if unrealized_profit >= 10:
                    trailing_stop_threshold = 1
                elif unrealized_profit >= 8:
                    trailing_stop_threshold = 2
                elif unrealized_profit >= 5:
                    trailing_stop_threshold = 3
                elif unrealized_profit >= 47:
                    trailing_stop_threshold = 4
                
            if new_price >= stop_loss_price:
                sell_profits[i] = entry_price - stop_loss_price
                break
                
            if j == max_look_ahead - 1:
                sell_profits[i] = entry_price - stop_loss_price
    
    # Add the results to the DataFrame
    df['buy_profit'] = buy_profits
    df['sell_profit'] = sell_profits
    
    return df


def simulate_trade_threaded(df):
    symbols = df['symbol'].unique()
    results = []
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(simulate_trade, df[df['symbol'] == symbol].copy()) for symbol in symbols]
        for future in futures:
            results.append(future.result())
    
    return pd.concat(results)