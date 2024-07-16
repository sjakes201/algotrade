from backtest import run_backtest
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

morning = {
    'model_path': './models/morning_model/xgboost_meta_model.pkl',
    'preprocessor_path': './models/morning_model/preprocessor.pkl',
    'data_path': './es_data/new/morning_testing_adjusted.csv'
}

midday = {
    'model_path': './models/midday_model/xgboost_meta_model.pkl',
    'preprocessor_path': './models/midday_model/preprocessor.pkl',
    'data_path': './es_data/new/midday_testing_adjusted.csv'
}

closing = {
    'model_path': './models/closing_model/closing_model.pkl',
    'preprocessor_path': './models/closing_model/closing_preprocessor.pkl',
    'data_path': './es_data/new/closing_testing_adjusted.csv'
}

model_path = closing['model_path']
preprocessor_path = closing['preprocessor_path']
data_path = closing['data_path']

def backtest_task(config, model_path, preprocessor_path, data_path, day_end_time):
    try:
        account = run_backtest(model_path, preprocessor_path, data_path, config, day_end_time)
        # profit = account.get_balance() - 10000
        # fees = account.get_total_fees()
        # return {
        #     'confidence_threshold': config['confidence_threshold'],
        #     'max_position_size': config['max_position_size'],
        #     'stop_loss_diff': config['stop_loss_diff'],
        #     'take_profit_diff': config['take_profit_diff'],
        #     'trailing_loss': config['trailing_loss'],
        #     'profit': profit,
        #     'fees': fees
        # }
        return account
    except Exception as e:
        print(f"Error in backtest_task with config {config}: {e}")
        return {
            'confidence_threshold': config['confidence_threshold'],
            'max_position_size': config['max_position_size'],
            'stop_loss_diff': config['stop_loss_diff'],
            'take_profit_diff': config['take_profit_diff'],
            'trailing_loss': config['trailing_loss'],
            'profit': None,
            'fees': None,
            'error': str(e)
        }

def main():
    configs = []

    for confidence_threshold in [0.75, 0.8, 0.85]:
        for max_position_size in [5, 6, 7]:
            for stop_loss_diff in [4,5,6,7]:
                for take_profit_diff in [8,10,12,15]:
                    for trailing_loss in [True, False]:
                        config = {
                            'confidence_threshold': confidence_threshold,
                            'max_position_size': max_position_size,
                            'stop_loss_diff': stop_loss_diff,
                            'take_profit_diff': take_profit_diff,
                            'trailing_loss': trailing_loss
                        }
                        configs.append(config)

    results = []

    # Limit the number of concurrent processes
    MAX_WORKERS = 8  # Adjust based on your CPU cores

    def process_batch(batch, start_index):
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(backtest_task, config, model_path, preprocessor_path, data_path, "16:00:00"): index for index, config in enumerate(batch, start=start_index)}
            for future in as_completed(futures):
                config_index = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Completed backtest with config {config_index}: {result}")
                except Exception as e:
                    print(f"Error in future result for config {config_index}: {e}")

    # Process the configs in batches to limit the number of processes
    BATCH_SIZE = MAX_WORKERS

    for i in range(0, len(configs), BATCH_SIZE):
        batch = configs[i:i + BATCH_SIZE]
        process_batch(batch, start_index=i)

    results_df = pd.DataFrame(results)
    results_df.to_csv('backtest_results_closing_1.csv', index=False)

    print("\nBacktesting completed and results saved csv")



if __name__ == '__main__':
    best_config = {
    'confidence_threshold': 0.8,
    'max_position_size': 7,
    'stop_loss_diff': 4,
    'take_profit_diff': 15,
    'trailing_loss': False
                }
    # main()
    print(backtest_task(best_config, model_path, preprocessor_path, data_path, "16:00:00"))
    
"""
Best configs:

Morning:
    'confidence_threshold': 0.85,
    'max_position_size': 7,
    'stop_loss_diff': 4,
    'take_profit_diff': 12,
    'trailing_loss': False

Midday: 
    'confidence_threshold': 0.85,
    'max_position_size': 7,
    'stop_loss_diff': 5,
    'take_profit_diff': 15,
    'trailing_loss': True
    
Closing: 
    'confidence_threshold': 0.8,
    'max_position_size': 7,
    'stop_loss_diff': 4,
    'take_profit_diff': 15,
    'trailing_loss': False
"""