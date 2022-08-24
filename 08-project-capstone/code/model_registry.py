import mlflow
from mlflow.tracking import MlflowClient

from search_run import mlflow_client

import pickle
from datetime import datetime

from prefect import flow, task
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule
from prefect.task_runners import SequentialTaskRunner


experiment_id = '1'
experiment_name = "customer_churn_prediction"
MLFLOW_TRACKING_URI = "sqlite:///customer_churn.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(experiment_name)

def get_best_model_info(exp_id: str, experiment_name: str):
    '''
    Query the best model from the mlflow database
    Return run ID and accuracy of the new best model from the experiments
    '''

    best_model = client.search_runs(
        experiment_ids=exp_id,
        filter_string="metrices.accuracy > 0.90",
        order_by=["metrics.accuracy.DESC"]
    )[0]

    best_run_id = best_model.info.run_id
    best_accuracy = best_model.data.metrics["accuracy"]

    for model_version in client.search_model_versions(f"name={experiment_name}"):        
        if model_version.run_id == best_run_id:
            version = model_version.version

    return best_run_id, best_accuracy, version


def transition_model_stage(MLFLOW_TRACKING_URI: str):
    '''
    Get the new model and model stage
    Put to the given stage
    '''

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    run_id




def get_current_model():
    '''
    Get the parameters of the current model in production
    Return run ID of the current model and accuracy
    '''



