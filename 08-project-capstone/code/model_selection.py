import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import pickle
import random
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
from prefect.task_runners import SequentialTaskRunner


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
def model_training(model, X_train, Y_train, x_val, y_val):
    '''
    Given the models list from the training session
    Then predict the validation set
    Return the metrices calculate from each model
    '''

    log_metrics = dict()

    churn = model.fit(X_train, Y_train)
    predition = churn.predict(x_val)
    accuracy = accuracy_score(predition, y_val)
    precision = precision_score(predition, y_val)
    recall = recall_score(predition, y_val)
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
    

    return log_metrics


@task
def save_model(models_dict, idx, model, std, model_namespace):
    '''
    Save the model pickle
    '''

    mlflow_path = Path("./mlflow/models")
    mlflow_path.mkdir(parents=True, exist_ok=True)

    with open(f'./mlflow/models/{model_namespace}.bin', 'wb') as f_out:
        pickle.dump(model, f_out)

    with open(f'./mlflow/models/{models_dict[idx]}-preprocessor.b', 'wb') as std_out:
                pickle.dump(std, std_out)
    print("Saved model successfully...")


@flow(task_runner=SequentialTaskRunner())
def model_experiment(dataset_path: str="../dataset/churn_train_dataset.csv"):
    '''
    Model experiment pipeline:
    1. Read dataframe
    2. Make label encoding
    3. Standardization
    4. Train and test split
    5. Model training
    6. Model evaluation
    '''

    models = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier(),
                GaussianNB(), SVC(), LGBMClassifier(), XGBClassifier(), GradientBoostingClassifier()]
    models_dict = {
        0: "LogisticRegrssion", 1: "DecissionTreeClassifier", 2: "KNeighborsClassifier",
        3: "RandomForestClassifier", 4: "GaussianNB", 5: "SVC", 6: "LGBMClassifier",
        7: "XGBClassifier", 8: "GradientBoostingClassifier"
    }


    for idx, model in enumerate(models):    

        with mlflow.start_run(run_name="model_selection"):

            test_size = .25
            random.seed(42)
            random_state = random.randint(1, 100)

            mlflow.set_tag("developer", "Surawut")
            mlflow.set_tag("Project", "Customer_Churn_Prediction")
            mlflow.set_tag("Model", models_dict[idx])
            mlflow.set_tag("dataset", dataset_path)
            mlflow.set_tag("Environment", "Dev")

            mlflow.log_param("random state", random_state)
            
            df = read_dataframe(dataset_path)
            df = label_encoding(df)
            df, std = prepare_features(df)
            X_train, x_val, Y_train, y_val = prepare_dataset(df, test_size, random_state)
            log_metrics = model_training(model, X_train, Y_train, x_val, y_val)
            mlflow.log_metrics(log_metrics)

            save_model(models_dict, idx, model, std, models_dict[idx])    
            mlflow.log_artifact(f'./mlflow/models/{models_dict[idx]}.bin', artifact_path='models_pickle')
            mlflow.log_artifact(f'./mlflow/models/{models_dict[idx]}-preprocessor.b', artifact_path='preprocessor')


if __name__ == "__main__":
    
    experiment_name = "customer_churn_prediction"
    MLFLOW_TRACKING_URI = "sqlite:///customer_churn.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    model_experiment()