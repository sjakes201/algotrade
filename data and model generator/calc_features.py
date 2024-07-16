import numpy as np

def market_regime(df):
    # Determine behavior based on ADX
    df['market_regime_ADX'] = np.where(df['ADX_50'] < 20, 'range-bound', 'trending')
    
    # Initialize the market_regime_MA column
    df['market_regime_MA'] = 'range-bound'
    
    # Determine behavior based on moving averages: 200, 100, 50, 20 SMA
    bull_condition = (df['sma_20'] > df['sma_50']) & (df['sma_50'] > df['sma_100']) & (df['sma_100'] > df['sma_200'])
    bear_condition = (df['sma_20'] < df['sma_50']) & (df['sma_50'] < df['sma_100']) & (df['sma_100'] < df['sma_200'])
    
    df.loc[bull_condition, 'market_regime_MA'] = 'bull'
    df.loc[bear_condition, 'market_regime_MA'] = 'bear'
    
    return df

def calculate_volatility(df, period=14):
    df['volatility'] = df['close'].rolling(window=period).std()
    return df

def calc_volume_ratio(df, window_size=14):
    df["volume_ma"] = df['volume'].rolling(window=window_size).mean()
    df["volume_ratio"] = df['volume'] / df["volume_ma"]
    
    return df

def add_total_signals(df):
    # Define the columns that contain the signals
    signal_columns = ['momentum_signal', 'mean_reversion_signal', 'breakout_signal', 'reversal_signal', 
                      'vwap_trend_signal', 'ema_12_26_signal', 'scalping_signal', 'heikin_ashi_signal']
    
    # Calculate the total number of 'BUY' and 'SELL' signals for each row
    df['total_buys'] = df[signal_columns].apply(lambda row: (row == 'BUY').sum(), axis=1)
    df['total_sells'] = df[signal_columns].apply(lambda row: (row == 'SELL').sum(), axis=1)
    
    return df