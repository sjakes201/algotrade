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