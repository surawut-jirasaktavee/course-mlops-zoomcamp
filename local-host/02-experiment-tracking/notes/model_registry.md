# MODEL REGISTRY
---
Reference [mlflow model registry docs](https://www.mlflow.org/docs/latest/registry.html)

![Model registry](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/local-host/02-experiment-tracking/images/model-registry-new.png)

ref: [image](https://databricks.com/blog/2019/10/17/introducing-the-mlflow-model-registry.html)


From the previous module in the tracking server and model management, you can store the model and model artifacts. Once you grow up your model and model artifact are the same as well. In this case, you can register those models into `MLflow` and you may decide that some of these models that ready for production, and then the deployment team can just take a look at the `Model Registry`.

## Model Registry

The `Model Registry` contains different stages:
* Staging
* Production
* Archive

Once the data scientist decides to put this model the production. The deployment team can just take a look at the `Model Registry` of that model and inspect what are the hyperparameters that were used what is the size of the model, and the performance, and move this model between the different stages.

The `Model Registry` help you in the case that you just deploy some model to the production and that model has the bad performance you can just move the old model from the archive stage(the stage after the production for the model that you didn't used it once you got the new model) and move them back to production.

The `MLflow` **dont' put your model to the production or move to other stages**. But just `Promoting` and `Demoting` model in registry to the diffenrent stages. You need to complement the model registry with some `CI/CD` tools.

## Interacting with MLflow with Tracking Client

In addition from the above story. Sometimes you may need to automate the process of `Registring`, `Promoting`, `Demoting` the models. In that case you need to use the `Tracking Client API` initialized as following:

```Python
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
```

From the code snippet above. Now we can use the client to interfact with MLflow backend as with the MLflow UI.
We can search for runs id by ascending order to filter the metric using The API:

```Python
from mlflow.entities import ViewType

runs = client.search_runs(
    experiment_ids = '<run_id>',
    filter_string = "metrics.rmse < <score>",
    run_view_type = ViewType.ACTIVE_ONLY,
    max_results = <ordering>,
    order_by = ['metric.rmse ASC']
)
```

And get information about the model selected from run_id resulting as rnumerator

```Python
for run in runs:
    print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['rmse']:.4f}")
```
![Model information]()

### Interacting with the Model Registry

Once we want to add the model to the model registry we can follow this way

```Python
mlflow.set_tracking_uri(mlflow_tracking_uri)

run_id = <your_run_id>"
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri=model_uri, name="taxi_trip_regressor")
```

And get the models in model registry with:

```Python
model_name = "taxi_trip_regressor"
latest_versions = client.get_latest_versions(name=model_name)

model_versions = list()

for version in latest_versions:
    print(f"version: {version.version}, stage: {version.current_stage}")
    model_versions.append(version.version)
```

For `Promoting` and `Demoting` the model

```Python
selected_model = "<model version>"
model_stages = ["Staging", "Production", "Archived", "None"]
new_stage = "<model stage>"

client.transition_model_version_stage(
    name=model_name,
    version=selected_model,
    stage=new_stage,
    archive_existing_versions=False
)
```

Sometimes we want to add some information or just note about what the step or activity

```Python
def update_model_metadata(model_name, new_prod_ver, old_prod_ver):
    
    from datetime import datetime
    date = datetime.today().date()
    
    
    client.update_model_version(
        name=model_name,
        version=new_prod_ver,
        description=f"Uploaded model to production on {date}")
    
    client.update_model_version(
        name=model_name,
        version=old_prod_ver,
        description=f"Transition model to archive on {date}")
    
    return f"Operation Succesfully"
```

Those code snippets above are just the example you can design your way.
