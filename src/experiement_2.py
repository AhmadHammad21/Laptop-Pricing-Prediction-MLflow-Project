import mlflow
from mlflow_utils import get_mlflow_experiment, log_regression_metrics_run
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from mlflow import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.models.signature import infer_signature
from mlflow.types.schema import Schema
from mlflow.types.schema import ParamSchema
from mlflow.types.schema import ParamSpec
from mlflow.types.schema import ColSpec

mlflow.autolog()
mlflow.xgboost.autolog()

if __name__ == "__main__":
    experiment = get_mlflow_experiment(experiment_name="Laptops Pricing Prediction")

    RANDOM_STATE = 42
    test_size = 0.2

    data = pd.read_csv("./data/final/experiment_1.csv")

    X = data.drop('Price_euros', axis=1)
    y = data['Price_euros']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    client = MlflowClient()


    with mlflow.start_run(experiment_id=experiment.experiment_id) as run_rf:
        
        model_rf = RandomForestRegressor(random_state=RANDOM_STATE)
        model_rf.fit(X_train, y_train)

        # Make predictions and evaluate RandomForest
        predictions_rf = model_rf.predict(X_test)
        train_predictions = model_rf.predict(X_train)

        log_regression_metrics_run(y_true=y_train, predictions=train_predictions, prefix="train", n_features=X_train.shape[1])    
        log_regression_metrics_run(y_true=y_test, predictions=predictions_rf, prefix="test", n_features=X_train.shape[1])    

        mlflow.sklearn.log_model(artifact_path="rf_model", sk_model=model_rf, registered_model_name="RandomForest")

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run_lr:
        cols_spec = []

        data_map = {
            'int64': 'integer',
            'float64': 'double',
            'bool': 'boolean',
            'str': 'string',
            "date": 'datetime'
        }

        for name, dtype in X_train.dtypes.to_dict().items():
            cols_spec.append(ColSpec(name=name, type=data_map[str(dtype)]))

        input_schema = Schema(inputs=cols_spec)
        output_schema = Schema([ColSpec(name="label", type="integer")])

        parameter = ParamSpec(name="lr_model", dtype="string", default="model1")
        param_schema = ParamSchema(params=[parameter])

        model_signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=param_schema)
        print("MODEL SIGNATURE")
        print(model_signature.to_dict())

        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)

        predictions_lr = model_lr.predict(X_test)
        train_predictions = model_lr.predict(X_train)

        log_regression_metrics_run(y_true=y_train, predictions=train_predictions, prefix="train", n_features=X_train.shape[1])    
        log_regression_metrics_run(y_true=y_test, predictions=predictions_lr, prefix="test", n_features=X_train.shape[1])    

        mlflow.sklearn.log_model(artifact_path="lr_model", sk_model=model_lr, registered_model_name="LinearRegression",
                                 signature=model_signature)
        client.update_model_version(name="LinearRegression", version=6, description="This is a Linear Regression model version")

        # client.transition_model_version_stage(name=model_name, version=1, stage="Staging")

        # # delete model version 
        # client.delete_model_version(name=model_name, version=1)
    

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run_lr:

        model_xgb = XGBRegressor()
        model_xgb.fit(X_train, y_train)

        # Make predictions and evaluate Linear Regression
        predictions_xgb = model_xgb.predict(X_test)
        train_predictions = model_xgb.predict(X_train)

        log_regression_metrics_run(y_true=y_train, predictions=train_predictions, prefix="train", n_features=X_train.shape[1])    
        log_regression_metrics_run(y_true=y_test, predictions=predictions_lr, prefix="test", n_features=X_train.shape[1])  

        mlflow.xgboost.log_model(xgb_model=model_xgb, artifact_path="xgboost_model", registered_model_name="Xgboost")
