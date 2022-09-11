import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import random
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from lightgbm import LGBMClassifier

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


@task
def read_dataframe(dataset_path: str):
    '''
    Read dataset from csv file and drop specific columns
    Return Pandas DataFrame
    '''
    df = pd.read_csv(dataset_path)
    print("Read dataset to Pandas DataFrame...")
   
    col_to_drop = ['State', 'Area code', 'Total day charge', #'International plan', 'Voice mail plan',
                'Total eve charge', 'Total night charge', 'Total intl charge']    
    df = df.drop(columns = col_to_drop, axis = 1)
    print("...Drop non use columns finish")

    return df


@task
def prepare_onehot(df):
    '''
    Get the dataframe and transform data from specific columns
    then return new dataframe with new values

    Change from 'No' -> False and 'Yes' -> True
    '''
    df['International plan'] = df['International plan'].replace({'No': False, 'Yes': True})
    df['Voice mail plan'] = df['Voice mail plan'].replace({'No': False, 'Yes': True})
    print("Prepare data to one hot encoding")

    return df


@task
def label_encoding(df):
    '''
    Label encoding for categorical feature in the dataset
    e.g. retuurn 1 when "Churn == True and 0 when "Churn" == False
    '''
    bin_cols = ['International plan', 'Voice mail plan', 'Churn']
    # bin_cols = ['Churn']
    le = LabelEncoder()
    for item in bin_cols:
        df[item] = le.fit_transform(df[item])

    print("Make label encoding for categorical feature")

    return df


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
def make_dict_victorizer(X_train, x_val):
    '''
    Uses dictionary vectorizer for processing the dictionaries
    '''
    dv = DictVectorizer()
    train_dicts = X_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    test_dicts = x_val.to_dict(orient='records')
    x_val = dv.fit_transform(test_dicts)

    return X_train, x_val, dv


@task
def model_training(model, X_train, Y_train, x_val, y_val):
    '''
    Given the models list from the training session
    Then predict the validation set
    Return the metrices calculate from each model
    '''

    log_metrics = dict()

    churn = model.fit(X_train, Y_train)
    prediction = churn.predict(x_val)
    accuracy = accuracy_score(prediction, y_val)
    precision = precision_score(prediction, y_val)
    recall = recall_score(prediction, y_val)
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
    

    return log_metrics


@task
def save_model(model, dv):
    '''
    Save the model pickle
    '''
    model_path = Path("./env/dev/models/")
    model_path.mkdir(parents=True, exist_ok=True)

    with open(f'./env/dev/models/model.pkl', 'wb') as f_out:
        pickle.dump((model, dv), f_out)

    print("Saved model successfully...")


@task
def send_model_service(model_source):
    '''
    Send best model from model storage to the service
    '''
    model_service_path = Path("./env/stage/models/")
    model_service_path.mkdir(parents=True, exist_ok=True)

    cmd = f"cp {model_source} {model_service_path}"
    os.system(cmd)
    print("Send model to service is complete")


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
    7. Save model
    '''

    model_version = 1
    model_stage = ["Staging", "Production"]
    experiment_id = 1
    experiment_name = "customer_churn_prediction"
    MLFLOW_TRACKING_URI = "sqlite:///customer_churn.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="model_training") as run:

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_name = f"customer_churn_prediction_model_lightgbm"

        test_size = .25
        random.seed(42)
        random_state = random.randint(1, 100)

        mlflow.set_tag("Project", "Customer_Churn_Prediction")
        mlflow.set_tag("Developer", "Surawut")
        mlflow.set_tag("Dataset", dataset_path)
        mlflow.set_tag("Model", "lightgbm")

        mlflow.log_param("random_state", random_state)
      
        df = read_dataframe(dataset_path)
        df = prepare_onehot(df)
        df = label_encoding(df)
        X_train, x_val, Y_train, y_val = prepare_dataset(df, test_size, random_state)
        X_train, x_val, dv = make_dict_victorizer(X_train, x_val)
        model = LGBMClassifier(n_estimators=100, n_jobs=1, random_state=random_state,
                                num_leaves=64, learning_rate=0.01)
        log_metrics = model_training(model, X_train, Y_train, x_val, y_val)
        mlflow.log_metrics(log_metrics)

        save_model(model, dv)

        mlflow.log_artifact(f'./env/dev/models/model.pkl', artifact_path='models')
        mlflow.register_model(model_uri=model_uri, name=model_name)
        
        mlflow.end_run()

    model_source = f"./env/dev/models/model.pkl"
    send_model_service(model_source)

    for model in client.search_model_versions(f"name='{model_name}'"):
        model_id = model.run_id
        if model_id == run_id:
            new_model_name = model.name
            new_model_version = model.version
            client.transition_model_version_stage(
                name=new_model_name,
                version=new_model_version,
                stage=model_stage[0],
                archive_existing_versions=False
            )

if __name__ == "__main__":

    model_experiment()

