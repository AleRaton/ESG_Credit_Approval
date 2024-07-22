import subprocess
import mlflow

def run_training(model_name, data_path, param_path):
    command = ['python', 'train_model.py', model_name, data_path, param_path]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

if __name__ == "__main__":
    data_path = "data/intermediate_results/prepared_data.csv"
    param_path = "models_parameters.json"
    
    # Add your models here
    models_to_train = ["random_forest", "logistic_regression"]
    
    for model in models_to_train:
        print(f"Training {model}...")
        run_training(model, data_path, param_path)