import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import pickle
from datetime import datetime

import os
from pathlib import Path

def get_current_model_info(model_name: str):
    '''
    Get the current model from Production
    Return ID and accuracy of current model
    '''
    for model_version in client.search_model_versions(f"name='{model_name}'"):
        model_stage = model_version.current_stage
        if (model_stage == "None") | (model_stage == "Archived"):
            continue
        elif (model_stage == "Production") | (model_stage == "Staging"):
            current_stage = model_version.current_stage
            current_model = model_version
            current_id = current_model.run_id
            current_version = model_version.version
            current_run = client.get_run(run_id=current_id)
            current_accuracy = current_run.data.metrics["accuracy"]
        else:
            current_id = 0
            current_accuracy = 0
            current_version = None

    print(f"[MLFLOW] Production model info...")
    print(f"[MLFLOW] model stage: {current_stage}")
    print(f"[MLFLOW] Run ID: {current_id}")
    print(f"[MLFLOW] Accuracy: {current_accuracy}")
    print(f"[MLFLOW] Version: {current_version}")
    
    return current_id, current_accuracy


def get_best_model_info(exp_id: str, model_name: str):
    '''
    Query the best model from the mlflow database
    Return run ID and accuracy of the new best model from the experiments
    '''

    best_model = client.search_runs(
        experiment_ids=exp_id,
        filter_string="metric.accuracy > 0.90",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metric.accuracy DESC"]
    )[0]

    best_run_id = best_model.info.run_id
    best_accuracy = best_model.data.metrics['accuracy']

    for model_version in client.search_model_versions(f"name='{model_name}'"):        
        if model_version.run_id == best_run_id:
            best_model_current_stage = model_version.current_stage
            best_model_name = model_version.name
            best_model_version = model_version.version
    print("[MLFLOW] Current model info...")
    print(f"[MLFLOW] model stage: {best_model_current_stage}")
    print(f"[MLFLOW] Run ID: {best_run_id}")
    print(f"[MLFLOW] Accuracy: {best_accuracy}")
    print(f"[MLFLOW] Version: {best_model_version}")

    return best_run_id, best_model_name, best_accuracy, best_model_version


def send_model_service(model_source):
    '''
    Send best model from model storage to the service
    '''
    model_service_path = Path("./churn_prediction_service/")
    model_service_path.mkdir(parents=True, exist_ok=True)

    cmd = f"cp {model_source} {model_service_path}"
    os.system(cmd)
    print("Send model to service is complete")


if __name__ == "__main__":

    experiment_id = 1
    model_name = "customer_churn_prediction_model_lightgbm"
    model_stage = ["Staging", "Production"]
    MLFLOW_TRACKING_URI = "sqlite:///customer_churn.db"
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    date = datetime.today().date()
    model_source = f"./env/stage/models/model.pkl"
    current_id, current_accuracy = get_current_model_info(model_name)
    print()
    best_run_id, best_model_name, best_accuracy, best_model_version = get_best_model_info(experiment_id, model_name)

    models = client.search_model_versions(f"name='{model_name}'")


    if (current_accuracy == best_accuracy) & (len(models) == 1) :
        client.transition_model_version_stage(
            name=best_model_name,
            version=best_model_version,
            stage=model_stage[1],
            archive_existing_versions=True
        )
        print("[MLFLOW] Transition best model to Production is complete")
        send_model_service(model_source)
    elif (current_accuracy < best_accuracy) & (len(models) > 1):
        client.transition_model_version_stage(
            name=best_model_name,
            version=best_model_version,
            stage=model_stage[1],
            archive_existing_versions=True
        )
        print("[MLFLOW] Transition best model to Production is complete")
        send_model_service(model_source)
    elif (current_accuracy >= best_accuracy) & (len(models) > 1):
        print("[MLFLOW] The current model is good enough | use the current model instead")

