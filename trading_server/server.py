import databento as db
import pandas as pd
import pytz
import joblib
from datetime import datetime, timedelta, timezone, time
from calc_data import calc_features
from use_model import get_prediction
from main import init_connection, process_signal
import traceback
from helpers import set_interval
import time as t
import sys
import threading

from dotenv import load_dotenv
import os
load_dotenv()

# Initialize the tradovate api connection and reconnect every 45 mins
init_connection()
set_interval(init_connection, 45 * 60)  # Reconnect every 45 mins

# Load in the morning and midday models and preprocessors
regular_models = {
    "morning": {
        "model": joblib.load('./models/morning_model/xgboost_meta_model.pkl'),
        "preprocessor": joblib.load('./models/morning_model/preprocessor.pkl'),
        "config": {
            'confidence_threshold': 0.85,
            'max_position_size': 7,
            'stop_loss_diff': 4,
            'take_profit_diff': 12,
            'trailing_loss': False
        }
    },
    "midday": {
        "model": joblib.load('./models/midday_model/xgboost_meta_model.pkl'),
        "preprocessor": joblib.load('./models/midday_model/preprocessor.pkl'),
        'config': {
            'confidence_threshold': 0.85,
            'max_position_size': 7,
            'stop_loss_diff': 5,
            'take_profit_diff': 15,
            'trailing_loss': True
        }
    },
    "closing": {
        "model": joblib.load('./models/closing_model/closing_model.pkl'),
        "preprocessor": joblib.load('./models/closing_model/closing_preprocessor.pkl'),
        'config': {
            'confidence_threshold': 0.8,
            'max_position_size': 7,
            'stop_loss_diff': 4,
            'take_profit_diff': 15,
            'trailing_loss': False
        }
    
    }
}

model_account_ids = {
    "demo_0": 10464470, 
    "demo_1": 10664702,
    "demo_2": 10744564,
    "demo_3": 10796357,
    "live": 370703
}


# Create an empty DataFrame with the expected columns
df = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "instrument_id"])
new_data = False
current_instrument_id = 118  # MESU4

program_start = datetime.now(timezone.utc)
print(f"Starting program at: {program_start}")


def predict_and_trade(df, account):    
    # Convert the last datetime to a Timestamp object and ensure it's in UTC
    utc_datetime = pd.Timestamp(df.iloc[-1]['datetime']).tz_localize('UTC')
    
    # Extract the time component
    utc_time = utc_datetime.time()
    print(f"Predicting for {utc_datetime}")
    
    models = regular_models
    
    # Determine which model and preprocessor to use based on the UTC time
    if time(13, 30) <= utc_time < time(15, 30):
        print(f"{utc_time} uses morning model")
        model = models["morning"]["model"]
        preprocessor = models["morning"]["preprocessor"]
        config = models["morning"]["config"]
    elif (time(15, 30) <= utc_time < time(18, 0)):
        print(f"{utc_time} uses midday model")
        model = models["midday"]["model"]
        preprocessor = models["midday"]["preprocessor"]
        config = models["midday"]["config"]
    elif (time(18, 0) <= utc_time < time(20, 0)):
        print(f"{utc_time} uses closing model")
        model = models["closing"]["model"]
        preprocessor = models["closing"]["preprocessor"]
        config = models["closing"]["config"]
    else:
        print(f"No prediction model for {utc_datetime}. Time: {utc_time}")
        return
    
    account_id = model_account_ids.get(account)
    confidence_level = config.get('confidence_threshold', 0.85)
    
    # Make a prediction
    prediction = get_prediction(df.iloc[[-1]], model, preprocessor, confidence_level)
    close_price = df.iloc[-1]['close']
    print(f"PREDICTION FOR {utc_datetime}, when price is {close_price} @ {confidence_level} confidence: {prediction}")
    if account_id is None:
        print(f"No account ID found for account type {account}. Either 'regular' or 'inverse' are valid account types.")
        return
    
    strategy = "bracket"
    process_signal(account_id, prediction, "MESU4", close_price, strategy=strategy, config=config, env='live')
    
    
def predict_all(df):
    # Create threads
    thread5 = threading.Thread(target=predict_and_trade, args=(df, "live"))

    # Start threads
    thread5.start()

    # Wait for both threads to complete
    thread5.join()
    
def update_data(data):
    try:
        if(data.instrument_id != current_instrument_id):
            return
        global df, new_data, program_start, calc_features
        
        # STEP 1: FORMAT AND APPEND NEW MINUTE DATA TO DATAFRAME
        utc_datetime = pd.to_datetime(data.ts_event / 1000000, unit='ms', utc=True)

        data_dict = {
            "datetime": utc_datetime,
            "open": data.open / 1e9,
            "high": data.high / 1e9,
            "low": data.low / 1e9,
            "close": data.close / 1e9,
            "volume": data.volume,
            "instrument_id": data.instrument_id,
        }
        row = pd.DataFrame([data_dict])
        # Append the row to the DataFrame
        df = pd.concat([df, row], ignore_index=True)
        
        if not new_data and utc_datetime > program_start:
            new_data = True
            print("Switching to new data!")
            print(f"Now running calc_features on a df with {len(df)} rows") 
            # initiate 'process all old data' function
            df = calc_features(df)
            predict_all(df)
        elif new_data:
            # calculate data for this one row   
            df = calc_features(df)
            predict_all(df)
        
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
    

# Create a live client and connect
live_client = db.Live(key=os.getenv("DB_KEY"))

# for real morning sessions, we need from 18:00 the previous day, since that's when VWAP calculations start
est = pytz.timezone('US/Eastern')
current_time_utc = datetime.now(pytz.utc)
current_time_est = current_time_utc.astimezone(est)
yesterday_1800_est = (current_time_est - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
start_time = yesterday_1800_est.astimezone(pytz.utc)

start_time = datetime.now(timezone.utc) - timedelta(minutes=400)

# Subscribe to the ohlcv-1s schema for a few symbols
live_client.subscribe(
    dataset="GLBX.MDP3",
    schema="ohlcv-1m",
    stype_in="continuous",
    symbols=["ES.c.0", "ES.c.1", "CL.c.0", "CL.c.1"],
    start=start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
)

# Add the callback function to handle incoming data
live_client.add_callback(update_data)

# Start streaming
live_client.start()

# Wait for X seconds before closing the connection
four_and_half_hours_in_seconds = 4.5 * 60 * 60

live_client.block_for_close(timeout=four_and_half_hours_in_seconds)

# Print the DataFrame to verify all data (optional)
# df.to_csv("ohlcv_data.csv", index=False)
