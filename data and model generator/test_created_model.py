import pandas as pd
import joblib
import numpy as np
import xgboost as xgb

# Load the model and preprocessor from the saved files
model = joblib.load('./model/midday_1130_200/xgboost_meta_model.pkl')
preprocessor = joblib.load('./model/midday_1130_200/preprocessor.pkl')

# Load the new data
new_df = pd.read_csv('./futures_data/testing/formatted/training_data_midday.csv')

# Drop unnecessary columns
X_new = new_df.drop(columns=['TRADE_SIGNAL', 'TRADE_WEIGHT', 'SUPPRESSED_WEIGHTS', 'symbol', 'datetime_est', 'volume', 'close', 'low', 'high', 'open', 'buy_profit', 'sell_profit'])

# Apply the saved preprocessing steps
X_new_processed = preprocessor.transform(X_new)

# Make probability predictions on the new data
probabilities = model.predict_proba(X_new_processed)

def predict_labels(confidence_threshold, probabilities):
    # Apply the threshold to get predictions
    predictions = []
    for prob in probabilities:
        if max(prob) < confidence_threshold:
            predictions.append('HOLD')
        else:
            predicted_class = np.argmax(prob)
            if predicted_class == 2:
                predictions.append('BUY')
            elif predicted_class == 0:
                predictions.append('SELL')
            else:
                predictions.append('HOLD')

    # Add the predictions to the new dataframe
    new_df[f'PREDICTION_{confidence_threshold}'] = predictions
    
predict_labels(0.5, probabilities)
predict_labels(0.6, probabilities)
predict_labels(0.7, probabilities)
predict_labels(0.8, probabilities)
predict_labels(0.9, probabilities)
# Save the results to a new CSV file
new_df.to_csv('./model/midday_1130_200/regular_predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")
