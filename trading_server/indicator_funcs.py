import pandas as pd
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


def calculate_momentum(df, period=14):
    df[f'momentum'] = df['close'] - df['close'].shift(period)
    return df

def calculate_wilders_rsi(df, window=14):
    # Calculate the price changes
    delta = df['close'].diff()

    # Separate the gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the initial average gain and loss
    avg_gain = gain.rolling(window=window, min_periods=window).mean().iloc[window-1]
    avg_loss = loss.rolling(window=window, min_periods=window).mean().iloc[window-1]

    # Create lists to hold the average gains and losses
    avg_gain_list = [avg_gain]
    avg_loss_list = [avg_loss]

    # Calculate the rest of the average gains and losses using the previous averages
    for i in range(window, len(df)):
        avg_gain = (avg_gain * (window - 1) + gain.iloc[i]) / window
        avg_loss = (avg_loss * (window - 1) + loss.iloc[i]) / window
        avg_gain_list.append(avg_gain)
        avg_loss_list.append(avg_loss)

    # Extend the initial values to align with the original dataframe
    avg_gain_series = pd.Series([np.nan] * (window - 1) + avg_gain_list, index=df.index)
    avg_loss_series = pd.Series([np.nan] * (window - 1) + avg_loss_list, index=df.index)

    # Calculate Relative Strength (RS)
    rs = avg_gain_series / avg_loss_series

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    # Handle any potential division by zero issues
    rsi[avg_loss_series == 0] = 100
    rsi[(avg_gain_series == 0) & (avg_loss_series == 0)] = 0

    # Assign RSI to a new column in the DataFrame
    df['rsi'] = rsi

    return df

def calculate_sma(df, period):
    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    return df

def calculate_ema(df, period):
    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df

def calculate_stochastic(df, k_period=14, d_period=14, slowing_period=3):
    # Calculate lowest low and highest high over the look-back period
    df['low_min'] = df['low'].rolling(window=k_period).min()
    df['high_max'] = df['high'].rolling(window=k_period).max()
    
    # Calculate %K
    df['stochastic_oscillator'] = 100 * ((df['close'] - df['low_min']) / (df['high_max'] - df['low_min']))
    
    # Smooth %K with the slowing period
    df['stochastic_oscillator_smooth'] = df['stochastic_oscillator'].rolling(window=slowing_period).mean()
    
    # Calculate %D as the simple moving average of the smoothed %K
    df['STOCH_SIGNAL'] = df['stochastic_oscillator_smooth'].rolling(window=d_period).mean()
    
    # Drop intermediate columns
    df.drop(columns=['low_min', 'high_max', 'stochastic_oscillator'], inplace=True)
    
    # Rename %K_smooth to %K
    df.rename(columns={'stochastic_oscillator_smooth': 'STOCH_OSCIL'}, inplace=True)
    
    return df

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    # Calculate the fast and slow EMAs
    df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()

    # Calculate the MACD
    df['macd'] = df['ema_fast'] - df['ema_slow']

    # Calculate the MACD Signal line
    df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()

    # Drop the intermediate columns to clean up the DataFrame
    df.drop(['ema_fast', 'ema_slow'], axis=1, inplace=True)

    return df

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    This function calculates the Bollinger Bands for a given DataFrame.
    
    Parameters:
    - df: DataFrame containing the 'close' prices.
    - period: The period for the moving average (default is 20).
    - std_dev: The number of standard deviations for the bands (default is 2).
    
    Returns:
    - df: DataFrame with added columns for Bollinger Bands (Upper, Middle, Lower).
    """
    df[f'bollinger_middle'] = df['close'].rolling(window=period).mean()
    df[f'bollinger_stddev'] = df['close'].rolling(window=period).std()
    
    df[f'bollinger_upper'] = df[f'bollinger_middle'] + (std_dev * df[f'bollinger_stddev'])
    df[f'bollinger_lower'] = df[f'bollinger_middle'] - (std_dev * df[f'bollinger_stddev'])

    # Drop the intermediate standard deviation column to clean up the DataFrame
    df.drop([f'bollinger_stddev'], axis=1, inplace=True)
    
    return df

def calculate_pivot_points(df):
    """
    Calculate pivot points and support/resistance levels.
    
    Parameters:
    - df: DataFrame containing the 'high', 'low', and 'close' prices.
    
    Returns:
    - df: DataFrame with added columns for pivot points and support/resistance levels.
    """
    df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    df['support1'] = (2 * df['pivot']) - df['high'].shift(1)
    df['resistance1'] = (2 * df['pivot']) - df['low'].shift(1)
    df['support2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))
    df['resistance2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
    return df

def calculate_historical_support_resistance(df, lookback=20):
    """
    Calculate historical support and resistance levels based on past highs and lows.
    
    Parameters:
    - df: DataFrame containing the 'high' and 'low' prices.
    - lookback: The number of periods to look back to identify support and resistance.
    
    Returns:
    - df: DataFrame with added columns for historical support and resistance levels.
    """
    df['historical_resistance'] = df['high'].rolling(window=lookback).max().shift(1)
    df['historical_support'] = df['low'].rolling(window=lookback).min().shift(1)
    return df


def calculate_adx(df, period=14):
    def wilder_smoothing(series, period):
        result = [series.iloc[:period].mean()]
        for value in series.iloc[period:]:
            result.append((result[-1] * (period - 1) + value) / period)
        return pd.Series(result, index=series.index[period - 1:])
    df['TR'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift()), 
                                     abs(df['low'] - df['close'].shift())))
    df['+DM'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']), 
                         np.maximum(df['high'] - df['high'].shift(), 0), 0)
    df['-DM'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()), 
                         np.maximum(df['low'].shift() - df['low'], 0), 0)
    
    df['TR_smooth'] = wilder_smoothing(df['TR'], period)
    df['+DM_smooth'] = wilder_smoothing(df['+DM'], period)
    df['-DM_smooth'] = wilder_smoothing(df['-DM'], period)

    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])
    
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))

    df[f'ADX_{period}'] = wilder_smoothing(df['DX'], period)
    return df


def calculate_vwap_1800_to_1800(df):
    def calculate_vwap_for_group(group):
        """
        Calculate the VWAP and its upper and lower bands (+/- 2 standard deviations) for a given group of data.

        Parameters:
        group (pd.DataFrame): DataFrame containing 'datetime', 'high', 'low', 'close', and 'volume' columns.

        Returns:
        pd.DataFrame: DataFrame with additional columns for VWAP, upper band, and lower band.
        """
        # Ensure the group is sorted by datetime to maintain order
        group['datetime'] = pd.to_datetime(group['datetime'], utc=True)
        group = group.sort_values(by='datetime')

        # Calculate typical price
        group['typical_price'] = (group['high'] + group['low'] + group['close']) / 3

        # Calculate cumulative volume and typical price * volume
        group['cumulative_volume'] = group['volume'].cumsum()
        group['cumulative_typical_price_volume'] = (group['typical_price'] * group['volume']).cumsum()

        # Calculate VWAP
        group['vwap'] = group['cumulative_typical_price_volume'] / group['cumulative_volume']

        # Calculate the VWAP standard deviation
        group['vwap_squared'] = ((group['typical_price'] - group['vwap']) ** 2 * group['volume']).cumsum()
        
        group['vwap_stddev'] = group.apply(lambda row: np.sqrt(row['vwap_squared'] / row['cumulative_volume']) if row['cumulative_volume'] != 0 else 0, axis=1)


        # Calculate upper and lower bands (+/-2 standard deviations)
        group['vwap_upper_band'] = group['vwap'] + 2 * group['vwap_stddev']
        group['vwap_lower_band'] = group['vwap'] - 2 * group['vwap_stddev']

        # Drop intermediate calculation columns
        group.drop(columns=['typical_price', 'cumulative_typical_price_volume', 'vwap_squared', 'cumulative_volume', 'vwap_stddev'], inplace=True)

        return group
    
    # Define the session start time
    session_start_time = pd.to_datetime("23:00").time()

    # Create a column to indicate if the session should increment
    df['session_increment'] = (df['datetime'].dt.time == session_start_time)

    # Create the vwap_session column by cumulative sum of session increments
    df['vwap_session'] = df['session_increment'].cumsum()
    df['vwap_session'] = df['vwap_session'].astype(str) + "_" + "session"
    
    # Apply VWAP calculation to each futures 'day' (18:00-18:00) labeled by the unique vwap_session ID
    df = df.groupby('vwap_session').apply(lambda x: calculate_vwap_for_group(x))
    df = df.reset_index(drop=True)

    return df


#### CANDLESTICK CALCULATIONS
def is_hammer(row):
        body = abs(row['close'] - row['open'])
        lower_shadow = min(row['open'], row['close']) - row['low']
        upper_shadow = row['high'] - max(row['open'], row['close'])
        return lower_shadow > 2 * body and upper_shadow <= body

def is_hanging_man(row):
    return is_hammer(row)  # Same logic as Hammer but in an uptrend

def is_bullish_engulfing(row1, row2):
    return (row1['close'] < row1['open'] and
            row2['close'] > row2['open'] and
            row2['close'] > row1['open'] and
            row2['open'] < row1['close'])

def is_bearish_engulfing(row1, row2):
    return (row1['close'] > row1['open'] and
            row2['close'] < row2['open'] and
            row2['close'] < row1['open'] and
            row2['open'] > row1['close'])

def is_doji(row, tolerance=0.001):
    return abs(row['open'] - row['close']) <= tolerance

def is_shooting_star(row):
    body = abs(row['close'] - row['open'])
    lower_shadow = min(row['open'], row['close']) - row['low']
    upper_shadow = row['high'] - max(row['open'], row['close'])
    return upper_shadow > 2 * body and lower_shadow <= body

def is_morning_star(row1, row2, row3):
    return (row1['close'] < row1['open'] and
            row2['close'] < row2['open'] and
            row3['close'] > row3['open'] and
            row2['open'] < row1['close'] and
            row2['close'] > row3['open'] and
            row3['close'] > (row1['open'] + row1['close']) / 2)

def is_evening_star(row1, row2, row3):
    return (row1['close'] > row1['open'] and
            row2['close'] < row2['open'] and
            row3['close'] < row3['open'] and
            row2['open'] > row1['close'] and
            row2['close'] < row3['open'] and
            row3['close'] < (row1['open'] + row1['close']) / 2)
    
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