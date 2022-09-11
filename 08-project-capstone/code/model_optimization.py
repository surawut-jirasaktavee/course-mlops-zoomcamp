import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import pickle
import random
import time
from datetime import datetime

from prefect import flow, task
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule
from prefect.task_runners import SequentialTaskRunner

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from lightgbm import LGBMClassifier

import hyperopt
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin


@task
def read_dataframe(dataset_path: str):
    '''
    Read dataset from csv file and drop specific columns
    Return Pandas DataFrame
    '''
    df = pd.read_csv(dataset_path)
    print("Read dataset to Pandas DataFrame...")
    
    col_to_drop = ['State', 'Area code', 'Total day charge', 'Total eve charge', 
               'Total night charge', 'Total intl charge']    
    df = df.drop(columns = col_to_drop, axis = 1)
    print("...Drop non use columns finish")

    return df


@task
def label_encoding(df):
    '''
    Label encoding for categorical feature in the dataset
    Return 1 when "Churn == True and 0 when "Churn" == False
    '''

    bin_cols = ['International plan', 'Voice mail plan', 'Churn']
    le = LabelEncoder()
    for item in bin_cols:
        df[item] = le.fit_transform(df[item])
    print("Make label encoding for categorical feature")

    return df


@task
def prepare_features(df):
    '''
    Standardize features for numerical columns then drop the original
    Return by merge with the new feature scaled
    '''

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
    print("Prepare numerical features finish")

    return df, std


@task
def prepare_dataset(df, test_size: float, random_state: int):
    '''
    Specify target column and define size of the test dataset with random state
    Return train and test dataset
    '''

    target_col = ['Churn']
    cols = [i for i in df.columns if i not in target_col]
    X_train, x_val, Y_train, y_val = train_test_split(df[cols], df[target_col], 
                                                        test_size=test_size, random_state=random_state,
                                                        shuffle=True)
    print("Prepard dataset is complete...")    

    return X_train, x_val, Y_train, y_val


@task
def optimization_model(X_train, x_val, Y_train, y_val, 
                        dataset_path: str="../dataset/churn_train_dataset.csv"):
    '''
    Use Hyperopt to tuning hyperparameters with parameters below
    Return status of optimization model
    '''

    def objective(params):

        with mlflow.start_run(run_name="hyper_params_tuning"):

            log_metrics = dict() 

            mlflow.set_tag("Project", "Customer_Churn_Prediction")
            mlflow.set_tag("Developer", "Surawut")
            mlflow.set_tag("Environment", "Dev")
            mlflow.set_tag("Dataset", dataset_path)
            mlflow.set_tag("Model", hyperopt)
            
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
                'seed': int(params['seed']),
                'objective': str(params['objective'])
            }

            model = LGBMClassifier(
                n_estimators=500,
                **params
            )

            model.fit(X_train, Y_train)
            prediction = model.predict(x_val)

            accuracy = accuracy_score(prediction, y_val)
            recall = recall_score(prediction, y_val)
            precision = precision_score(prediction, y_val)
            f1 = f1_score(prediction, y_val)

            print('############################################')
            print('Algorithm:', type(model).__name__)
            print('Accuracy Score', '{:.4f}'.format(accuracy))
            print('Precision Score', '{:.4f}'.format(precision))
            print('Recal score', '{:.4f}'.format(recall))
            print('F1 score', '{:.4f}'.format(f1))
            print("Evaluated model successfully...")
            print('############################################\n')

            log_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

            mlflow.log_params(params)
            mlflow.log_metrics(log_metrics)
            mlflow.end_run()

            print("Hyperparameters tuning is complete")

        return {'accuracy': accuracy, 'status': STATUS_OK}
        
    search_space = {
        'num_leaves': hp.uniform('num_leaves', 8, 64),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
        'min_data_in_leaf': hp.uniform('min_data_in_leaf', 0, 6),
        'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
        'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
        'seed': 42,
        'objective': 'binary'
    }

    best = fmin(fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=100,
                # catch_eval_exceptions=False,
                trials=Trials(),
                verbose=1
    )

    return


@task
def get_best_params(experiment_id: str):
    '''
    Query the best parameters from Hyperopt with accuracy
    More than 90% and return parameters dictionary
    '''
    
    params = dict()   
    hyperopt = client.search_runs(
            experiment_ids=experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            filter_string="metric.accuracy > 0.90 and tags.model == 'hyperopt'",
            order_by=["metric.accuracy.DESC"]
        )[0]

    params['num_leaves'] = hyperopt.data.prams['num_leaves']
    params['colsample_bytree'] = hyperopt.data.params['colsample_bytree']
    params['learning_rate'] = hyperopt.data.params['learning_rate']
    params['subsample_for_bin'] = hyperopt.data.params['subsample_for_bin']
    params['feature_fraction'] = hyperopt.data.params['feature_fraction']
    params['bagging_fraction'] = hyperopt.data.params['bagging_fraction']
    params['min_data_in_leaf'] = hyperopt.data.params['min_data_in_leaf']
    params['lambda_l1'] = hyperopt.data.params['lambda_l1']
    params['lambda_l2'] = hyperopt.data.params['lambda_l2'] 
    params['seed'] = hyperopt.data.params['seed']

    return params


@task
def train_best_model(X_train, x_val, Y_train, y_val, params):
    '''
    Train the lightgbm model with hyperparameters from hyperopt
    Return model and metrics evaluation
    '''

    log_metrics = dict()

    best_params = {
        'num_leaves': params['num_leaves'],
        'colsample_bytree': params['colsample_bytree'],
        'learning_rate': params['learning_rate'],
        'subsample_for_bin': params['subsample_for_bin'],
        'feature_fraction': params['feature_fraction'],
        'bagging_fraction': params['bagging_fraction'],
        'min_data_in_leaf': params['min_data_in_leaf'],
        'lambda_l1': params['lambda_l1'],
        'lambda_l2': params['lambda_l2'],
        'seed': params['seed']
    }


    model = LGBMClassifier(best_params).fit(X_train, Y_train)
    predition = model.predict(x_val)
    accuracy = accuracy_score(predition, y_val)
    recall = recall_score(predition, y_val)
    precision = precision_score(predition, y_val)
    f1 = f1_score(predition, y_val)


    print('############################################')
    print('Algorithm:', type(model).__name__)
    print('Accuracy Score', '{:.4f}'.format(accuracy))
    print('Precision Score', '{:.4f}'.format(precision))
    print('Recal score', '{:.4f}'.format(recall))
    print('F1 score', '{:.4f}'.format(f1))
    print("Evaluated model successfully...")
    print('############################################\n')

    log_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return model, log_metrics


@flow(task_runner=SequentialTaskRunner())
def model_optimization(dataset_path: str="../dataset/churn_train_dataset.csv"):

    model_name = "churn_prediction-model"
    experiment_id = '1'

    test_size = .25
    random.seed(42)
    random_state = random.randint(1, 100)
        
    df = read_dataframe(dataset_path)
    df = label_encoding(df)
    df, std = prepare_features(df)
    X_train, x_val, Y_train, y_val = prepare_dataset(df, test_size, random_state)
    optimization_model(X_train, x_val, Y_train, y_val)

    # with mlflow.start_run(run_name="model_optimization"):
        
    #     mlflow.set_tag("Project", "Customer_Churn_Prediction")
    #     mlflow.set_tag("Developer", "Surawut")
    #     mlflow.set_tag("Environment", "Dev")
    #     mlflow.set_tag("Dataset", dataset_path)
    #     mlflow.set_tag("Model", "LGBMClassifier")

        # params = get_best_params(experiment_id)
        # model, log_metrics = train_best_model(X_train, x_val, Y_train, y_val, params)

        # mlflow.log_param("random_state", random_state)
        # mlflow.log_params(params)
        # mlflow.log_metrics(log_metrics)
        # mlflow.lightgbm.log_model(model, artifact_path="model")
        # mlflow.end_run()



if __name__ == "__main__":

    experiment_name = "customer_churn_prediction"
    MLFLOW_TRACKING_URI = "sqlite:///customer_churn.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    model_optimization()