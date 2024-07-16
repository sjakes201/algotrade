from calc_candlesticks import is_hammer, is_bullish_engulfing, is_bearish_engulfing, is_doji, is_hanging_man, is_morning_star, is_evening_star, is_shooting_star
import pandas as pd
import numpy as np
"""
This is a momentum trading strategy that uses the following indicators:
[RSI, MACD, MACD signal, Momentum indicator, 50 SMA]
"""
def momentum_trader(df):
    signals = []
    for i, row in df.iterrows():
        if (row['close'] > row['sma_50'] and row['rsi'] < 70 and
            row['macd'] > row['macd_signal'] and row['momentum'] > 0):
            signals.append("BUY")
        elif (row['close'] < row['sma_50'] and row['rsi'] > 30 and
              row['macd'] < row['macd_signal'] and row['momentum'] < 0):
            signals.append("SELL")
        else:
            signals.append("HOLD")
    df['momentum_signal'] = signals
    return df

"""
This is a mean reversion trading strategy that uses the following indicators:
[Bollinger Bands (Upper, Lower), RSI]
"""
def mean_reversion_trader(df):
    signals = []
    for i, row in df.iterrows():
        if row['close'] < row['bollinger_lower'] and row['rsi'] < 30:
            signals.append("BUY")
        elif row['close'] > row['bollinger_upper'] and row['rsi'] > 70:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    df['mean_reversion_signal'] = signals
    return df

"""
This is a breakout trading strategy that uses the following indicators:
[resistance (calc from pivots), support (calc from pivots), historical resistance (period low), historical support (period high), 50 sma, 50 ema, volume ratio]
"""
def breakout_trader(df):
    signals = []
    for i, row in df.iterrows():
        if row['volume_ratio'] > 1.5:
            if row['close'] > row['historical_resistance']:
                signals.append("BUY")
            elif row['close'] > row['resistance1']:
                signals.append("BUY")
            elif row['close'] > row['sma_50'] or row['close'] > row['ema_50']:
                signals.append("BUY")
            elif row['close'] < row['historical_support']:
                signals.append("SELL")
            elif row['close'] < row['support1']:
                signals.append("SELL")
            elif row['close'] < row['sma_50'] or row['close'] < row['ema_50']:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        else:
            signals.append("HOLD")
    df['breakout_signal'] = signals
    return df


"""
This is a reversal trading strategy that uses the following indicators:
[RSI, MACD, MACD signal, Candlestick Patterns, ADX, DI+, DI-]
"""
def reversal_trader(df):
    macd_gap = 0.1
    
    signals = []
    for i in range(2, len(df)):
        if pd.isna(df.iloc[i]['ADX_14']):
            signals.append(np.nan)
            continue
        row1, row2, row3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        signal = "HOLD"
        
        if df.iloc[i]['ADX_14'] > 25:  # Check for strong trend
            trend_direction = "UP" if df.iloc[i]['+DI'] > df.iloc[i]['-DI'] else "DOWN"
            
            if trend_direction == "UP":
                if is_bearish_engulfing(row1, row2) and row3['rsi'] > 70:
                    signal = "SELL"
                elif is_hanging_man(row3) and row3['rsi'] > 70:
                    signal = "SELL"
                elif is_shooting_star(row3) and row3['rsi'] > 70:
                    signal = "SELL"
                elif is_doji(row2) and row3['close'] < row2['close']:  # Confirming Doji with bearish next candle
                    signal = "SELL"
                elif is_evening_star(row1, row2, row3) and row3['rsi'] > 70:
                    signal = "SELL"
            elif trend_direction == "DOWN":
                if is_bullish_engulfing(row1, row2) and row3['rsi'] < 30:
                    signal = "BUY"
                elif is_hammer(row3) and row3['rsi'] < 30:
                    signal = "BUY"
                elif is_morning_star(row1, row2, row3) and row3['rsi'] < 30:
                    signal = "BUY"
                elif is_doji(row2) and row3['close'] > row2['close']:  # Confirming Doji with bullish next candle
                    signal = "BUY"
                
            if signal == "BUY" and row3['macd'] < (row3['macd_signal'] - macd_gap):
                signal = "HOLD"  # Nullify the buy signal if MACD does not confirm
            elif signal == "SELL" and row3['macd'] > (row3['macd_signal'] + macd_gap):
                signal = "HOLD"  # Nullify the sell signal if MACD does not confirm
                
        signals.append(signal)
    # Use 2 None values because the first two candles can't properly analyze all candlestick patterns
    df['reversal_signal'] = [np.nan, np.nan] + signals
    return df

"""
This is range trading strategy using VWAP as support and resistance levels. It uses 5 period EMA reversion, a spike in volume, and RSI to confirm the respecting of 
of a VWAP defined support or resistance line
"""
def vwap_trend_trader(df):
    """
    Three support / resistance lines: vwap_upper_band, vwap, and vwap_lower_band
    We analyze the three most recent candles
    
    Identify what band you are closest to
    Identify in what direction this would bounce if it bounces
    
    If the close is within 15% of the gap between the two closest bands
    and if the most recent candle and least recent candle's emas form a bounce with the middle ema (the middle ema is closer to the line than the other two)
    and if volume ratio > 2
    and if rsi < 30 (bounce up) or rsi > 70 (bounce down)
    
    Then trade
    
    Avenus for variation: RSI, volume ratio, tolerance?
    """
    results = ["HOLD"] 

    for i in range(1, len(df)):        
        previous_close = df.iloc[i-1]['close']
        current_close = df.iloc[i]['close']
        # If it's outside the VWAP bands, it's not range trading
        if (df.iloc[i]['close'] < df.iloc[i]['vwap_lower_band'] or 
            df.iloc[i]['close'] > df.iloc[i]['vwap_upper_band']):
            results.append("HOLD")
            continue
        
        # Find the closest VWAP band
        closest_vwap = ["vwap_upper_band", abs(df.iloc[i]['vwap_upper_band'] - current_close)]
        
        middle_band_distance = abs(df.iloc[i]['vwap'] - current_close)
        if middle_band_distance < closest_vwap[1]:
            closest_vwap = ["vwap", middle_band_distance]
            
        lower_band_distance = abs(df.iloc[i]['vwap_lower_band'] - current_close)
        if lower_band_distance < closest_vwap[1]:
            closest_vwap = ["vwap_lower_band", lower_band_distance]

        # Determine bounce direction
        bouncing_direction = "UP"
        if closest_vwap[0] == "vwap_upper_band" or (closest_vwap[0] == "vwap" and current_close < df.iloc[i]['vwap']):
            bouncing_direction = "DOWN"
        
        # Calculate the gap between bands and tolerance
        if current_close > df.iloc[i]['vwap']:
            band_gap = df.iloc[i]['vwap_upper_band'] - df.iloc[i]['vwap']
        else:
            band_gap = df.iloc[i]['vwap'] - df.iloc[i]['vwap_lower_band']
        
        tolerance = band_gap * 0.125
        if bouncing_direction == "UP":
            distance = df.iloc[i]['close'] - df.iloc[i][closest_vwap[0]]
        else:
            distance = df.iloc[i][closest_vwap[0]] - df.iloc[i]['close']

        # Check proximity to VWAP band
        if abs(distance) > tolerance or distance < 0:
            results.append("HOLD")
            continue
            
        # Check EMA bounce pattern
        bouncing_emas = False
        if bouncing_direction == "UP":
            if current_close > previous_close:
                bouncing_emas = True
        elif bouncing_direction == "DOWN":
            if current_close < previous_close:
                bouncing_emas = True

        if not bouncing_emas:
            results.append("HOLD")
            continue
        
        # Check for volume ratio spike. The spike that tests the line should have high volume, and high volume on move-away spike is also a 
        # good sign (less important). Weigh them accordingly
        if (0.25 * df.iloc[i]['volume_ratio'] + 0.75* df.iloc[i-1]['volume_ratio']) < 1:
            results.append("HOLD")
            continue

        # Check RSI bounds
        if bouncing_direction == "UP" and df.iloc[i]['rsi'] > 40:
            results.append("HOLD")
            continue
        elif bouncing_direction == "DOWN" and df.iloc[i]['rsi'] < 60:
            results.append("HOLD")
            continue

        # Generate trading signal
        if bouncing_direction == "UP":
            results.append("BUY")
        else:
            results.append("SELL")
    df["vwap_trend_signal"] = results
    return df

"""
This is a standard EMA crossover trader. It uses
[12 ema, 26 ema, 50 ema, volume ratio]

When the 12 crosses the 26, it considers a signal to trade in the direction of the crossover
The close price must be on the correct side of the 50 ema (above if crossing over and below if crossing under)
There must be a volume spike (ratio > 1.5)
If this is all satisfied, it will signal a trade. Else hold

Avenues for variation: volume ratio, recent crossovers
"""

def ema_trader_12_26(df):
    # Find every row with a new crossover
    df["12_over_26"] = df["ema_12"] > df["ema_26"]
    df["crossover"] = df["12_over_26"] != df["12_over_26"].shift()
    df['recent_crossovers'] = df['crossover'].rolling(window=15, min_periods=1).sum()

    # Initialize everything to HOLD
    results = ["HOLD"] * len(df)
    
    # Look for signal criteria
    for i in range(1, len(df)):
        ema_distance = abs(df.iloc[i]["ema_12"] - df.iloc[i]["ema_26"])
        if (df.iloc[i]["crossover"] or (i > 1 and df.iloc[i-1]["crossover"])) and (ema_distance > 0.1) and (df.iloc[i]['recent_crossovers'] < 2):
            if df.iloc[i]["12_over_26"]:
                # crossing upwards
                if df.iloc[i]['close'] > df.iloc[i]['ema_50'] and df.iloc[i]['volume_ratio'] > 1:
                    results[i] = 'BUY'
            else:
                # crossing downwards
                if df.iloc[i]['close'] < df.iloc[i]['ema_50'] and df.iloc[i]['volume_ratio'] > 1:
                    results[i] = 'SELL'
            
    df["ema_12_26_signal"] = results
    # Clean up temporary columns
    df.drop(columns=['12_over_26', 'crossover', 'recent_crossovers'], inplace=True)
    return df
    
    
def scalping_trader(df):
    signals = []
    for i in range(1, len(df)):
        if df.iloc[i]['close'] > df.iloc[i]['vwap'] and df.iloc[i]['volume_ratio'] >= 1.5:
            signal = "BUY"
        elif df.iloc[i]['close'] < df.iloc[i]['vwap'] and df.iloc[i]['volume_ratio'] >= 1.5:
            signal = "SELL"
        else:
            signal = "HOLD"
        signals.append(signal)
    df['scalping_signal'] = [None] + signals
    return df

def heikin_ashi_trader(df):
    signals = []
    ha_open = (df.iloc[0]['open'] + df.iloc[0]['close']) / 2  # Initial Heikin Ashi Open
    
    for i in range(len(df)):
        ha_close = (df.iloc[i]['open'] + df.iloc[i]['high'] + df.iloc[i]['low'] + df.iloc[i]['close']) / 4
        if i == 0:
            ha_open = (df.iloc[0]['open'] + df.iloc[0]['close']) / 2  # Initial Heikin Ashi Open
        else:
            ha_open = (ha_open + ha_close) / 2
        
        ha_high = max(df.iloc[i]['high'], ha_open, ha_close)
        ha_low = min(df.iloc[i]['low'], ha_open, ha_close)
        
        if ha_close > ha_open:
            signal = "BUY"
        elif ha_close < ha_open:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        signals.append(signal)
    
    df['heikin_ashi_signal'] = signals
    return df
