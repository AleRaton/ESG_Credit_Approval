import json
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
import os
from datetime import datetime
experiment_name = "ESG"
current_config_path = 'configs/XGBoost_ESG.json'  

# Set the base directory for MLflow tracking logs
tracking_dir = "/workspaces/ESG_Credit_Approval/mlflowlogs/mlruns"
# Set a separate directory for artifacts
artifact_dir = "/workspaces/ESG_Credit_Approval/mlflowlogs/artifacts"

mlflow.set_tracking_uri(f"file://{tracking_dir}")

with open(current_config_path, 'r') as f:
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
    X = pd.DataFrame(X, columns=features)

# Data splitting
test_size = config['data_split']['test_size']
random_state = config['data_split']['random_state']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Model selection
model_type = config['model']['type']
hyperparameters = config['model']['hyperparameters']

if model_type == 'RandomForestRegressor':
    model = RandomForestRegressor(**hyperparameters)
elif model_type == 'SVR':
    model = SVR(**hyperparameters)
elif model_type == 'XGBRegressor':
    model = XGBRegressor(**hyperparameters)
elif model_type == 'Ridge':
    model = Ridge(**hyperparameters)
else:
    raise ValueError(f"Model type {model_type} is not supported.")

# Enable autologging
mlflow.sklearn.autolog()

# Define experiment name
mlflow.set_experiment(experiment_name)

# Create a unique run name
run_name = f"{model_type} run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

with mlflow.start_run(run_name=run_name):
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    
    # Log parameter, metrics, and model
    mlflow.log_param("model_type", model_type)
    mlflow.log_params(hyperparameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Log the model in the "model" directory
    mlflow.sklearn.log_model(model, "model")

    # Create directories for artifacts within the current run's artifact directory
    run_artifact_dir = mlflow.get_artifact_uri()
    eda_dir = os.path.join(run_artifact_dir, "EDA")
    boxplot_dir = os.path.join(eda_dir, "boxplots")
    os.makedirs(boxplot_dir, exist_ok=True)

    # Log config.json
    config_path = os.path.join(run_artifact_dir, f"{model_type}.json")
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file)
    mlflow.log_artifact(config_path, "config")

    # Generate and log boxplots
    for column in X.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=X[column])
        plt.title(f'Boxplot of {column}')
        boxplot_path = os.path.join(boxplot_dir, f"{column}_boxplot.png")
        plt.savefig(boxplot_path)
        plt.close()
        mlflow.log_artifact(boxplot_path, "boxplots")
    
    # Generate and log correlation matrix
    corr = X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    corr_path = os.path.join(eda_dir, 'correlation_matrix.png')
    plt.savefig(corr_path)
    plt.close()
    mlflow.log_artifact(corr_path, "EDA")

print("Done logging to MLflow")