# EXPERIMENT TRACKING WITH MLFLOW
---

## MLflow Tracking Client API

An interface that used to automate processes is the Tracking API that we need to initialized through:

```Python
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI
```

The `Client` is an object that managing `experiment`, `runs`, `models`, and `model register`

## Creating experiments

The Python API that use to create the new experiments is `client.create_experiment("experiment_name")`
let's initialize the mlflow experiment with the code:

```Python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("experment_name") # use your experiment name
```

Once we created the experiment the experiment ID with automatically generated start by 1. because everyone get default experiment with experiment ID: 0 since you use mlflow.

From my notebook we can then track a run of this code snippet:

```Python
with mlflow.start_run(): # initialize the experiment run
  mlflow.set_tag("developer", " ") # you can add any name that you want
  
  # log path of dataset
  mlflow.log_param("train-path", "add-your-train-path-here")
  mlflow.log_param("val-parth", "add-your-val-path-here")
  
  alpha = 0.01
  mlflow.log_param("alpha", alpha) # log parameter
  

  lr = Lasso(alpha)
  lr.fit(X_train, y_train)

  y_pred = lr.predict(X_val)
  rmse = mean_squared_error(y_val, y_pred, squared=False)
  mlflow.log_metric("rmse", rmse) # log your metric here
```

