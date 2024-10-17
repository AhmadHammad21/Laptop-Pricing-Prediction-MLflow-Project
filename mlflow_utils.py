import mlflow
import mlflow.entities
from typing import Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def create_mlflow_experiment(experiment_name: str, artifact_location: str, tags: dict[str, Any]) -> str:

    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except:
        print(f"expriment: {experiment_name} already there")
        experiment_id = mlflow.get_experiment_by_name(name=experiment_name)

    return experiment_id
        

def get_mlflow_experiment(experiment_id: str=None, experiment_name: str=None) -> mlflow.entities.Experiment:

    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError("No such experiment id or name")
    
    return experiment


def log_regression_metrics_run(y_true, predictions, prefix="test", n_features=None):
    """
    Logs multiple regression metrics for the current MLflow run.
    
    Args:
    y_true (array-like): True target values.
    predictions (array-like): Predicted target values.
    prefix (str): Prefix for the metric name (e.g., 'train', 'test').
    n_features (int): Number of features used in the model (required for Adjusted R2).
    """
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, predictions)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100

    # R-squared (R2)
    r2 = r2_score(y_true, predictions)

    # Adjusted R-squared (Adjusted R2)
    if n_features is not None:
        n_samples = len(y_true)
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    else:
        adj_r2 = None  # Adjusted R2 requires number of features
    
    # Log metrics to MLflow
    mlflow.log_metric(f"{prefix}_mean_absolute_error", mae)
    mlflow.log_metric(f"{prefix}_root_mean_squared_error", rmse)
    mlflow.log_metric(f"{prefix}_mean_absolute_percentage_error", mape)
    mlflow.log_metric(f"{prefix}_r2", r2)
    
    # Log adjusted R2 if n_features is provided
    if adj_r2 is not None:
        mlflow.log_metric(f"{prefix}_adjusted_r2", adj_r2)
