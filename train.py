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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from a file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# Load and preprocess data
def load_data(data_path, features, target_variable, imputation_config):
    df = pd.read_csv(data_path)
    X = df[features]
    y = df[target_variable]

    if imputation_config['enabled']:
        strategy = imputation_config['strategy']
        imputer = SimpleImputer(strategy=strategy)
        X = imputer.fit_transform(X)
        X = pd.DataFrame(X, columns=features)
    
    return X, y

# Split data
def split_data(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Select model
def get_model(model_type, hyperparameters):
    if model_type == 'RandomForestRegressor':
        return RandomForestRegressor(**hyperparameters)
    elif model_type == 'SVR':
        return SVR(**hyperparameters)
    elif model_type == 'XGBRegressor':
        return XGBRegressor(**hyperparameters)
    elif model_type == 'Ridge':
        return Ridge(**hyperparameters)
    else:
        raise ValueError(f"Model type {model_type} is not supported.")

# Train and log model
def train_and_log_model(model, X_train, X_test, y_train, y_test, config, run_name):
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        r2 = r2_score(y_test, predictions)
        
        mlflow.log_param("model_type", config['model']['type'])
        mlflow.log_params(config['model']['hyperparameters'])
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        run_artifact_dir = mlflow.get_artifact_uri()
        eda_dir = os.path.join(run_artifact_dir, "EDA")
        boxplot_dir = os.path.join(eda_dir, "boxplots")
        os.makedirs(boxplot_dir, exist_ok=True)

        config_path = os.path.join(run_artifact_dir, f"{config['model']['type']}.json")
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file)
        mlflow.log_artifact(config_path, "config")

        for column in X_train.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=X_train[column])
            plt.title(f'Boxplot of {column}')
            boxplot_path = os.path.join(boxplot_dir, f"{column}_boxplot.png")
            plt.savefig(boxplot_path)
            plt.close()
            mlflow.log_artifact(boxplot_path, "boxplots")
        
        corr = X_train.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        corr_path = os.path.join(eda_dir, 'correlation_matrix.png')
        plt.savefig(corr_path)
        plt.close()
        mlflow.log_artifact(corr_path, "EDA")

# Main execution
def main():
    experiment_name = "NO ESG"
    current_config_path = 'model_configs/XGBoost.json'  
    data_path = 'data/processed_data/cleaned_data.csv'
    mlflow_dir = "/workspaces/ESG_Credit_Approval/mlflow"

    mlflow.set_tracking_uri(mlflow_dir)
    
    config = load_config(current_config_path)
    
    features = config['feature_selection']['features_to_keep']
    target_variable = config['target_variable']
    X, y = load_data(data_path, features, target_variable, config['missing_value_imputation'])
    
    X_train, X_test, y_train, y_test = split_data(X, y, config['data_split']['test_size'], config['data_split']['random_state'])
    
    model = get_model(config['model']['type'], config['model']['hyperparameters'])
    
    mlflow.sklearn.autolog()
    mlflow.set_experiment(experiment_name)
    
    run_name = f"{config['model']['type']} run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    train_and_log_model(model, X_train, X_test, y_train, y_test, config, run_name)

if __name__ == "__main__":
    try:
        main()
        logging.info("Done logging to MLflow")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
