import pandas as pd
import numpy as np

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
        group (pd.DataFrame): DataFrame containing 'datetime_est', 'high', 'low', 'close', and 'volume' columns.

        Returns:
        pd.DataFrame: DataFrame with additional columns for VWAP, upper band, and lower band.
        """
        # Ensure the group is sorted by datetime_est to maintain order
        group = group.sort_values(by='datetime_est')

        # Calculate typical price
        group['typical_price'] = (group['high'] + group['low'] + group['close']) / 3

        # Calculate cumulative volume and typical price * volume
        group['cumulative_volume'] = group['volume'].cumsum()
        group['cumulative_typical_price_volume'] = (group['typical_price'] * group['volume']).cumsum()

        # Calculate VWAP
        group['vwap'] = group['cumulative_typical_price_volume'] / group['cumulative_volume']

        # Calculate the VWAP standard deviation
        group['vwap_squared'] = ((group['typical_price'] - group['vwap']) ** 2 * group['volume']).cumsum()
        group['vwap_stddev'] = np.sqrt(group['vwap_squared'] / group['cumulative_volume'])

        # Calculate upper and lower bands (+/-2 standard deviations)
        group['vwap_upper_band'] = group['vwap'] + 2 * group['vwap_stddev']
        group['vwap_lower_band'] = group['vwap'] - 2 * group['vwap_stddev']

        # Drop intermediate calculation columns
        group.drop(columns=['typical_price', 'cumulative_typical_price_volume', 'vwap_squared', 'cumulative_volume', 'vwap_stddev'], inplace=True)

        return group
    
    df['datetime_est'] = pd.to_datetime(df['datetime_est'])

    # Define the session start time
    session_start_time = pd.to_datetime("18:00").time()

    # Create a column to indicate if the session should increment
    df['session_increment'] = (df['datetime_est'].dt.time == session_start_time)
    # MANUAL CORRECTION FOR PRIMARY 1 YEAR DATA SET ONLY: we are missing 17-18 in the data here and need to manually set vwap reset
    df.loc[df['datetime_est'] == "2023-11-12 19:00:00", 'session_increment'] = True

    # Create the vwap_session column by cumulative sum of session increments
    df['vwap_session'] = df['session_increment'].cumsum()
    df['vwap_session'] = df['vwap_session'].astype(str) + "_" + df['symbol']
    
    # Apply VWAP calculation to each futures 'day' (18:00-18:00) labeled by the unique vwap_session ID
    df = df.groupby('vwap_session').apply(lambda x: calculate_vwap_for_group(x))

    return df