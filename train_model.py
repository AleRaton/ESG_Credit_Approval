import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def load_data(file_path):
    return pd.read_csv(file_path)

def load_parameters(file_path):
    with open(file_path) as f:
        return json.load(f)

def train_random_forest(X_train, y_train, params):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train, params):
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model

def main(model_name, data_path, param_path):
    df = load_data(data_path)
    params = load_parameters(param_path)
    
    X = df.drop("target_column", axis=1)
    y = df["target_column"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_name == "random_forest":
        model = train_random_forest(X_train, y_train, params[model_name])
    elif model_name == "logistic_regression":
        model = train_logistic_regression(X_train, y_train, params[model_name])
    
    accuracy = model.score(X_test, y_test)
    
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params[model_name])
        mlflow.sklearn.log_model(model, model_name + "_model")
        mlflow.log_metric("accuracy", accuracy)

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1]
    data_path = sys.argv[2]
    param_path = sys.argv[3]
    main(model_name, data_path, param_path)