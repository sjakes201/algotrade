import joblib
from Account import Account
import pandas as pd

def run_backtest(model_path, preprocessor_path, data_path, config, day_end_time):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    data = pd.read_csv(data_path)

    algo_params = {
        'confidence_threshold': config['confidence_threshold'],
        "max_position_size": config['max_position_size'],
        'stop_loss_diff': config['stop_loss_diff'],
        'take_profit_diff': config['take_profit_diff'],
        "trailing_loss": config['trailing_loss']
    }

    starting_balance = 10000
    acc = Account("Test", starting_balance, algo_params)
    acc.set_model(model, preprocessor)
    acc.set_contract_config(per_contract_fee=0.35, point_dollar_value=5, points_slippage=0)

    for i in range(len(data)):
        candle_row = data.iloc[[i]]
        datetime_est = candle_row['datetime_est'].values[0]
        time = datetime_est.split(" ")[1]
        if time == day_end_time:
            close = candle_row['close'].values[0]
            symbol = candle_row['symbol'].values[0]
            acc.close_all_positions(symbol, close)
            acc.log_day()
        else:
            acc.process_recent_candle(candle_row)

    return acc