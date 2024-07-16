import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import optuna
import joblib

# Load the CSV file into a DataFrame
df = pd.read_csv('./futures_data/1_year/formatted/training_data_closing.csv')

# Preprocess data
X = df.drop(columns=['TRADE_SIGNAL', 'TRADE_WEIGHT', 'SUPPRESSED_WEIGHTS', 'symbol', 'datetime_est', 'volume', 'close', 'low', 'high', 'open', 'buy_profit', 'sell_profit'])
y = df['TRADE_SIGNAL']

# Split data into categorial and numeric feature types
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
numerical_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Convert the categorical features to one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

# Extract feature names
num_features = numerical_features
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_features = np.concatenate([num_features, cat_features])

# Mapping target labels to numeric
y_numeric = y.map({'BUY': 2, 'HOLD': 1, 'SELL': 0})

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y_numeric)

# Compute sample weights to address class imbalance on the resampled data
class_weights_resampled = compute_sample_weight(class_weight='balanced', y=y_resampled)
sample_weights_resampled = class_weights_resampled

# Best parameters from grid search 
best_params = {'learning_rate': 0.1944155455656647, 'max_depth': 9, 'n_estimators': 269, 'colsample_bytree': 0.8984508754225905, 'subsample': 0.888839870342431}
# Best parameters from grid search SHORT TERM MODEL
# best_params = {'learning_rate': 0.11149228747371114, 'max_depth': 9, 'n_estimators': 269, 'colsample_bytree': 0.8860305390252818, 'subsample': 0.6421507311691057}


# Initialize the XGBoost classifier with the best parameters
model = xgb.XGBClassifier(**best_params)

# Evaluate the model using K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_classification_reports = []
confusion_matrices = []

for train_index, test_index in kf.split(X_resampled):
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]
    weights_train, weights_test = sample_weights_resampled[train_index], sample_weights_resampled[test_index]

    model.fit(X_train, y_train, sample_weight=weights_train)

    y_pred = model.predict(X_test)
    y_pred_labels = pd.Series(y_pred).map({2: 'BUY', 1: 'HOLD', 0: 'SELL'})
    y_test_labels = pd.Series(y_test).map({2: 'BUY', 1: 'HOLD', 0: 'SELL'})

    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    fold_accuracies.append(accuracy)
    fold_classification_reports.append(classification_report(y_test_labels, y_pred_labels, output_dict=True))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=['BUY', 'HOLD', 'SELL'])
    confusion_matrices.append(cm)

avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
print(f'Average Accuracy: {avg_accuracy}')

def feature_importance():
    # Plot feature importance
    importance = model.feature_importances_
    if len(all_features) != len(importance):
        raise ValueError("Mismatch between the number of features and the length of importance array.")
    importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)

    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()
    
def aggregate_report():
    # Aggregate classification report
    aggregate_report = {
        'BUY': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'HOLD': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'SELL': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'accuracy': [],
        'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'weighted avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    }

    for report in fold_classification_reports:
        for key in report:
            if isinstance(report[key], dict):
                for metric in report[key]:
                    aggregate_report[key][metric].append(report[key][metric])
            else:
                aggregate_report[key].append(report[key])

    for key in aggregate_report:
        if isinstance(aggregate_report[key], dict):
            for metric in aggregate_report[key]:
                aggregate_report[key][metric] = np.mean(aggregate_report[key][metric])
        else:
            aggregate_report[key] = np.mean(aggregate_report[key])

    print(aggregate_report)

def display_cm():
    # Average confusion matrix
    avg_cm = np.mean(confusion_matrices, axis=0)
    print(avg_cm)
    # Set print options to suppress scientific notation
    np.set_printoptions(suppress=True, precision=0)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm, display_labels=['BUY', 'HOLD', 'SELL'])
    disp.plot(cmap='Blues')
    plt.show()

def hyperparameter_search_optuna(X, y, sample_weights):
    def objective(trial):
        # Define the parameter search space
        param = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 0.9),
            'subsample': trial.suggest_uniform('subsample', 0.5, 0.9)
        }

        # Initialize the XGBoost classifier with the trial parameters
        model = xgb.XGBClassifier(**param)
        
        # Perform cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            weights_train, weights_test = sample_weights[train_index], sample_weights[test_index]

            model.fit(X_train, y_train, sample_weight=weights_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # Return the mean accuracy
        return np.mean(accuracies)

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1)  # Use all available cores

    # Print the best parameters found
    best_params = study.best_params
    print("Best parameters found: ", best_params)

    # Save the best parameters to a text file
    with open('best_params.txt', 'w') as f:
        f.write(f"Best parameters found: {best_params}\n")

    return best_params

# best_params = hyperparameter_search_optuna(X_resampled, y_resampled, sample_weights_resampled)
# print("Best parameters found: ", best_params)

def save_model():
    joblib.dump(model, 'closing_model.pkl')
    joblib.dump(preprocessor, 'closing_preprocessor.pkl')


feature_importance()
aggregate_report()
display_cm()
save_model()
