import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import random
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import mlflow

from prefect import flow, task
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

dataset = "./dataset/churn_train_dataset.csv"
models = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier(),
        GaussianNB(), SVC(), LGBMClassifier(), XGBClassifier(), GradientBoostingClassifier()]
n_classes = 2

def read_dataframe(dataset):

    df = pd.read_csv(dataset)
    col_to_drop = ['State', 'Area code', 'Total day charge', 'Total eve charge', 
               'Total night charge', 'Total intl charge']
    bin_cols = ['International plan', 'Voice mail plan', 'Churn']
    df = df.drop(columns = col_to_drop, axis = 1)

    le = LabelEncoder()
    for item in bin_cols:
        df[item] = le.fit_transform(df[item])

    return df

def prepare_features(df):

    cate_cols = ['International plan', 'Voice mail plan']
    num_cols = ['Account length', 'Number vmail messages', 'Total day minutes',
                'Total day calls', 'Total eve minutes', 'Total eve calls',
                'Total night minutes', 'Total night calls', 'Total intl minutes',
                'Total intl calls', 'Customer service calls']

    std = StandardScaler()
    df_scaled = std.fit_transform(df[num_cols])
    df_scaled = pd.DataFrame(df_scaled, columns=num_cols)
    df = df.drop(columns = num_cols, axis = 1)
    df = df.merge(df_scaled, left_index=True, right_index=True, how = "left")

    return df, std

def prepare_dataset(df):

    target_col = ['Churn']
    cols = [i for i in df.columns if i not in target_col]
    
    # Split train and test set
    X_train, x_val, Y_train, y_val = train_test_split(df[cols], df[target_col], 
                                                        test_size = .25, random_state = 42)
    
    return X_train, x_val, Y_train, y_val
    

def churn_prediction_model(models, X_train, x_val, Y_train, y_val):
    
    model_results = dict()

    # Model training and prediction
    for model in models:
        model.fit(X_train, Y_train)
        predition = model.predict(x_val)
            
        print('Algorithm:', type(model).__name__)
        accuracy = accuracy_score(y_val, predition)
        print(f"Accuracy Score:, {accuracy}\n")
    

        model_results[model] = accuracy
        
    return model_results


def train_model_search(X_train, x_val, Y_train, y_val):

    import hyperopt
    from hyperopt import hp, tpe, Trials
    from hyperopt.fmin import fmin

    def objective(params):
        # with mlflow.start_run():
        params = {
            'num_leaves': int(params['num_leaves']),
            'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
            'learning_rate': '{:.3f}'.format(params['learning_rate']),
            'subsample_for_bin': int(params['subsample_for_bin']),
            'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
            'bagging_fraction': '{:.3f}'.format(params['bagging_fraction']),
            # 'min_data_in_leaf': int(params['min_data_in_leaf']),
            'lambda_l1': int(params['lambda_l1']),
            'lambda_l2': int(params['lambda_l2']),
            'seed': int(params['seed']),
            'objective': str(params['objective'])
        }
    
        lgbm = LGBMClassifier(
            n_estimators=500,
            **params
        )

        lgbm.fit(X_train, Y_train)
        pred = lgbm.predict(x_val)
        accuracy = accuracy_score(pred, y_val)
        print("Accuracy {:.3f}".format(accuracy))

        return accuracy
        
    search_space = {
        'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
        'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
        'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
        'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
        'seed': 42
    }

    if n_classes > 2:
        search_space['objective'] = "multiclass"
    else:
        search_space['objective'] = "binary"

    best = fmin(fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=100,
                catch_eval_exceptions=False,
                trials=Trials(),
                verbose= 1
    )


        # mlflow.log_metric("accuracy", accuracy)
        # mlflow.xgboost.log_model(booster, artifact_path="model")

def main(models: list(), dataset: str="./dataset/churn_train_dataset.csv"):

    df = read_dataframe(dataset)
    df, std = prepare_features(df)
    X_train, x_val, Y_train, y_val = prepare_dataset(df)
    model_results = churn_prediction_model(models, X_train, x_val, Y_train, y_val)
    train_model_search(X_train, x_val, Y_train, y_val)

if __name__ == "__main__":
    main(models)