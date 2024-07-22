import json
import pandas as pd
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

mlflow.set_tracking_uri("file///tmp/mlruns")  # You can choose any directory you have write access to


# Load configuration
with open('configs/RandomForest.json', 'r') as f:
    config = json.load(f)

# Load CSV data
df = pd.read_csv('data/processed_data/cleaned_data.csv')

# Feature selection
features = config['feature_selection']['features_to_keep']
X = df[features]
y = df[config['target_variable']]

# Missing value imputation
if config['missing_value_imputation']['enabled']:
    strategy = config['missing_value_imputation']['strategy']
    imputer = SimpleImputer(strategy=strategy)
    X = imputer.fit_transform(X)

# Data splitting
test_size = config['data_split']['test_size']
random_state = config['data_split']['random_state']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Model selection
model_type = config['model']['type']
hyperparameters = config['model']['hyperparameters']

if model_type == 'RandomForestRegressor':
    model = RandomForestRegressor(**hyperparameters)
elif model_type == 'LinearRegression':
    model = LinearRegression(**hyperparameters)
elif model_type == 'SVR':
    model = SVR(**hyperparameters)
elif model_type == 'XGBRegressor':
    model = XGBRegressor(**hyperparameters)
else:
    raise ValueError(f"Model type {model_type} is not supported.")

# Enable autologging
mlflow.sklearn.autolog()

with mlflow.start_run():
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Log parameter, metrics, and model
    mlflow.log_param("model_type", model_type)
    mlflow.log_params(hyperparameters)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

print("Done logging to MLflow")
