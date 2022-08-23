import pandas as pd
import numpy as np

import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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

from search_run import mlflow_client

import hyperopt
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin

from prefect import flow, task
from prefect.filesystems import S3

from prefect.task_runners import SequentialTaskRunner
from prefect.orion.schemas.schedules import IntervalSchedule

dataset_path = "./dataset/churn_train_dataset.csv"
models = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier(),
        GaussianNB(), SVC(), LGBMClassifier(), XGBClassifier(), GradientBoostingClassifier()]

"""bash
export AWS_ACCESS_KEY_ID="ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="SECRET_ACCESS_KEY"
export MLFLOW_TRACKING_URI="ec2-13-57-28-199.us-west-1.compute.amazonaws.com"
export EXPERIMENT_NAME="churn_telecom_prediction"
export BUCKET_NAME="mlflow-artifacts-remote-prem"
"""

n_classes = 2
BUCKET_NAME = os.getenv("BUCKET_NAME")
os.environ["AWS_PROFILE"] = "default"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")

    
# @task
def read_dataframe(dataset_path: str):

    df = pd.read_csv(dataset_path)
    col_to_drop = ['State', 'Area code', 'Total day charge', 'Total eve charge', 
               'Total night charge', 'Total intl charge']
    bin_cols = ['International plan', 'Voice mail plan', 'Churn']
    df = df.drop(columns = col_to_drop, axis = 1)

    le = LabelEncoder()
    for item in bin_cols:
        df[item] = le.fit_transform(df[item])

    return df


# @task
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


# @task
def prepare_dataset(df):

    target_col = ['Churn']
    cols = [i for i in df.columns if i not in target_col]
    
    # Split train and test set
    X_train, x_val, Y_train, y_val = train_test_split(df[cols], df[target_col], 
                                                        test_size = .25, random_state = 42)
    
    return X_train, x_val, Y_train, y_val
    

# @task
def churn_prediction_model(models, X_train, x_val, Y_train, y_val):
    
    model_results = dict()

    # Model training and prediction
    for model in models:
        with mlflow.start_run(run_name="model_selection"):
            
            mlflow.set_tag("developer", "surawut")
            mlflow.set_tag("model", model)

            model.fit(X_train, Y_train)
            predition = model.predict(x_val)
                
            print('Algorithm:', type(model).__name__)
            accuracy = accuracy_score(y_val, predition)
            print(f"Accuracy Score:, {accuracy}\n")

            mlflow.log_metric("accuracy", accuracy)    

            model_results[model] = accuracy
        
    return model_results


def train_model_search(X_train, x_val, Y_train, y_val):

    def objective(params):
        # with mlflow.start_run(run_name="hyper_params_tuning"):

            # mlflow.set_tag("model", hyperopt)
            
        params = {
            'num_leaves': int(params['num_leaves']),
            'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
            'learning_rate': '{:.3f}'.format(params['learning_rate']),
            'subsample_for_bin': int(params['subsample_for_bin']),
            'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
            'bagging_fraction': '{:.3f}'.format(params['bagging_fraction']),
            'min_data_in_leaf': int(params['min_data_in_leaf']),
            'lambda_l1': int(params['lambda_l1']),
            'lambda_l2': int(params['lambda_l2']),
            # 'seed': int(params['seed'])
            # 'objective:': str(params['objective'])
        }
    
        lgbm = LGBMClassifier(
            n_estimators=500,
            **params
        )

        lgbm.fit(X_train, Y_train)
        pred = lgbm.predict(x_val)
        accuracy = accuracy_score(pred, y_val)
            # mlflow.log_metric("accuracy", accuracy)
            # mlflow.log_params(params)
            # mlflow.lightgbm.log_model(lgbm, artifact_path="model")
        print("Accuracy {:.3f}".format(accuracy))

        return {'accuracy': accuracy, 'status': STATUS_OK}
        
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
        # 'seed': 42
    }

    # if n_classes > 2:
    #     search_space['objective'] = "multiclass"
    # else:
    #     search_space['objective'] = "binary"

    best = fmin(fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=100,
                catch_eval_exceptions=False,
                trials=Trials(),
                verbose=1
    )

# @task
# def lightgbm_param(MLFLOW_TRACKING_URI, exp_num: int=1):
 
#     client = mlflow_client(mlflow_tracking_uri=MLFLOW_TRACKING_URI , exp_num_ID=exp_num)
#     run = client.runs()

#     params = dict()    

#     params['num_leaves'] = run.data.params['num_leaves']
#     params['colsample_bytree'] = run.data.params['colsample_bytree']
#     params['learning_rate'] = run.data.params['learning_rate']
#     params['subsample_for_bin'] = run.data.params['subsample_for_bin']
#     params['feature_fraction'] = run.data.params['feature_faction']
#     params['bagging_fraction'] = run.data.params['bagging_fraction']
#     params['min_data_in_leaf'] = run.data.params['min_data_in_leaf']
#     params['lambda_l1'] = run.data.params['lambda_l1']
#     params['lambda_l2'] = run.data.params['lambda_l2']
#     params['objective'] = run.data.params['objective']
#     params['seed'] = run.data.params['seed']

#     return params

# @task
# def train_best_model(X_train, x_val, Y_train, y_val, params):
#     with mlflow.start_run(run_name="model_optimization"):

#         best_params = {
#             'num_leaves': params['num_leaves'],
#             'colsample_bytree': params['colsample_bytree'],
#             'learning_rate': params['learning_rate'],
#             'subsample_for_bin': params['subsample_for_bin'],
#             'feature_fraction': params['feature_fraction'],
#             'bagging_fraction': params['bagging_fraction'],
#             'min_data_in_leaf': params['min_data_in_leaf'],
#             'lambda_l1': params['lambda_l1'],
#             'lambda_l2': params['lambda_l2'],
#             'objective': params['objective'],
#             'seed': params['seed']
#         }

#         mlflow.log_params(best_params)

#         model = LGBMClassifier().fit(X_train, Y_train)
#         predition = model.predict(x_val)
#         accuracy = accuracy_score(y_val, predition)
#         print(f"Accuracy Score:, {accuracy}\n")

#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.lightgbm.log_model(lgbm, artifact_path="model")
#     return accuracy
        
        

# @flow(name="model_experiment",
#     version="v1",
#     task_runner=SequentialTaskRunner(),
#     retries=3)
def main(dataset_path: str="./dataset/churn_train_dataset.csv"):
    
    mlflow.set_tracking_uri(f"http://{MLFLOW_TRACKING_URI}:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = read_dataframe(dataset_path)
    df, std = prepare_features(df)
    X_train, x_val, Y_train, y_val = prepare_dataset(df)
    model_results = churn_prediction_model(models, X_train, x_val, Y_train, y_val)
    train_model_search(X_train, x_val, Y_train, y_val)
    # params = lightgbm_param(MLFLOW_TRACKING_URI, exp_num=1)
    # print(params)
    # train_best_model(X_train, x_val, Y_train, y_val, params)



if __name__ == "__main__":
    main()