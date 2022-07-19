import pandas as pd
import pickle

from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow

from search_run import mlflow_client

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("taxi_trip_prediction-experiment")

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

def add_features(train_path="./data/green_tripdata_2021-01.parquet",
                 val_path="./data/green_tripdata_2021-02.parquet"):
    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)

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


def param():
    
    client = mlflow_client()
    run = client.runs()
    learning_rate = run.data.params['learning_rate']
    max_depth = run.data.params['max_depth']
    min_child_weight = run.data.params['min_child_weight']
    objective = run.data.params['objective']
    reg_alpha = run.data.params['reg_alpha']
    reg_lambda = run.data.params['reg_lambda']
    seed = run.data.params['seed']

    return learning_rate, max_depth, min_child_weight, objective, reg_alpha, reg_lambda, seed

def train_best_model(train, valid, y_val, dv, learning_rate, max_depth, min_child_weight, objective, reg_alpha, reg_lambda, seed):
   
    with mlflow.start_run():
        
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
       
        # Config hyperparameters from hyper-opt
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
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        artifact_path = Path('./mlflow/models/')
        artifact_path.mkdir(parents=True, exist_ok=True)

        with open(f"{artifact_path}/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(f"{artifact_path}/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, dv = add_features()
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    learning_rate, max_depth, min_child_weight, objective, reg_alpha, reg_lambda, seed = param()
    # train_model_search(train, valid, y_val)
    train_best_model(train, valid, y_val, dv, learning_rate, max_depth, min_child_weight, objective, reg_alpha, reg_lambda, seed)