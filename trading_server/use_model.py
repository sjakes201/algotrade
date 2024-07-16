import pandas as pd
import numpy as np

feature_columns = ['momentum_signal', 'mean_reversion_signal', 'breakout_signal',
       'reversal_signal', 'vwap_trend_signal', 'ema_12_26_signal',
       'scalping_signal', 'heikin_ashi_signal', 'market_regime_ADX',
       'market_regime_MA', 'volatility', 'total_buys', 'total_sells',
       'momentum_signal_buy_accum', 'momentum_signal_sell_accum',
       'mean_reversion_signal_buy_accum', 'mean_reversion_signal_sell_accum',
       'breakout_signal_buy_accum', 'breakout_signal_sell_accum',
       'reversal_signal_buy_accum', 'reversal_signal_sell_accum',
       'vwap_trend_signal_buy_accum', 'vwap_trend_signal_sell_accum',
       'ema_12_26_signal_buy_accum', 'ema_12_26_signal_sell_accum',
       'scalping_signal_buy_accum', 'scalping_signal_sell_accum',
       'heikin_ashi_signal_buy_accum', 'heikin_ashi_signal_sell_accum']

def determine_label(probabilities, confidence_threshold):
    # Apply the threshold to get predictions
    if max(probabilities) < confidence_threshold:
        return 'HOLD'
    else:
        predicted_class = np.argmax(probabilities)
        if predicted_class == 2:
            return 'BUY'
        elif predicted_class == 0:
            return 'SELL'
        else:
            return 'HOLD'

    # Add the predictions to the new dataframe

def get_prediction(df, model, preprocessor, confidence_threshold = 0.5):
    global feature_columns
    prepared_df = df.loc[:, feature_columns]
    X_processed = preprocessor.transform(prepared_df)
    probabilities = model.predict_proba(X_processed)
    answer = determine_label(probabilities[0], confidence_threshold)
    print(f"get_prediction() called. Generated probabilities are: {probabilities}, confidence threshold is {confidence_threshold}, and the answer is {answer}")
    return answer
