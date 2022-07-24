import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow

from pathlib import Path

from prefect.deployments import FlowScript
from prefect.deployments import Deployment
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.filesystems import RemoteFileSystem
from prefect.logging import get_run_logger
from prefect.packaging import FilePackager
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient 

@task
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

@task
def add_features(df_train, df_val):
    # df_train = read_dataframe(train_path)
    # df_val = read_dataframe(val_path)

    print(len(df_train))
    print(len(df_val))

    df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
    df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

    categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv

@task
def train_model_search(train, valid, y_val):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,
        trials=Trials()
    )
    return

@task
def param(id=1):
    
    mlflow_tracking_uri = 'sqlite:///mlflow.db'
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    experiments = client.list_experiments()
    exp_ids = list(experiments[id].experiment_id)[-1]
    
    run = client().search_runs(
                experiment_ids=exp_ids,
                filter_string="",
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=10,
                order_by=None  
    )
    
    learning_rate = run.data.params['learning_rate']
    max_depth = run.data.params['max_depth']
    min_child_weight = run.data.params['min_child_weight']
    objective = run.data.params['objective']
    reg_alpha = run.data.params['reg_alpha']
    reg_lambda = run.data.params['reg_lambda']
    seed = run.data.params['seed']

    return learning_rate, max_depth, min_child_weight, objective, reg_alpha, reg_lambda, seed

@task
def train_best_model(train, valid, y_val, dv, learning_rate, max_depth, min_child_weight, objective, reg_alpha, reg_lambda, seed):
    with mlflow.start_run():

        best_params = {
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'objective': objective,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'seed': seed
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        
        artifact_path = Path('./models/')
        artifact_path.mkdir(parents=True, exist_ok=True)

        with open(f"{artifact_path}/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(f"{artifact_path}/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

@flow(task_runner=SequentialTaskRunner())
def main(train_path: str="./data/green_tripdata_2021-01.parquet",
        val_path: str="./data/green_tripdata_2021-02.parquet"):
    
    from search_run import mlflow_client
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    X_train = read_dataframe(train_path)
    X_val = read_dataframe(val_path)
    X_train, X_val, y_train, y_val, dv = add_features(X_train, X_val).result()
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    train_model_search(train, valid, y_val)
    learning_rate, max_depth, min_child_weight, objective, reg_alpha, reg_lambda, seed = param()
    train_best_model(train, valid, y_val, dv, learning_rate, max_depth, min_child_weight, objective, reg_alpha, reg_lambda, seed)

aws_s3_file_packager = FilePackager(filesystem=RemoteFileSystem(
    basepath="s3://prefect-artifacts-prem/artifacts/",
    settings={
        "key": ""        "secret": "",
    }
))

Deployment(
    flow=FlowScript(path="./prefect_deploy.py", name="main"),
    name="model_training",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    packager=aws_s3_file_packager,
    tags=["test module3"]
)

# if __name__ == "main":
#     main()


# PREFECT_API_URL="http://13.52.76.201:4200/api"