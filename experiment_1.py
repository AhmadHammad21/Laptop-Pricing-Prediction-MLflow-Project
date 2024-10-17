import mlflow
from mlflow_utils import create_mlflow_experiment


if __name__ == "__main__":
    experiment_id = create_mlflow_experiment(
        experiment_name="predict_pricing",
        artifact_location="predict_pricing",
        tags={"env": "dev", "version": "1.0.0"}
    )

    print("experiment id", experiment_id)

    with mlflow.start_run() as run:

        mlflow.log_param("learnign_rate", 0.01)
