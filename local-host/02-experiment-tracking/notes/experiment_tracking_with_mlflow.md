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

## Hyperparameter Optimization Tracking

In my case from the experiment with `XGBoost` model I have used `hyperopt` the optimization framework. Then you can also wrapping the `heperopt` objective inside a mlflow tracking with the same as the code above. we can track every optimization run that was ran by `hyperopt`. So we can log the parameters passed by `hyperopt` as we log the metric as below:

```Python
import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)

def objective(params):
    with mlflow.start_run(run_name="fine tuning"):
        mlflow.set_tag("developer", "surawut")
        mlflow.set_tag("model", "hyperopt")
        mlflow.log_params(params)
        
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        
    return {'loss': rmse, 'status': STATUS_OK}

search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}

best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=10,
    trials=Trials()
)
```

As the code above, First import the libraries and then define the train and validate dataset and create the function then I wrap the training and validation block inside `with mlflow.start_run()` and log the parameters using `log_params` from the `search_space` and I have added more tag about the model or optimization and you can define the name of that run by the parameter `run_name="your run name"` and finally I also log the metric with rmse.

## Autologging

In addition way, we can logging the parameters by the `Autologging feature` in MLflow. There are two ways.

MLflow:

```Python
mlflow.autolog()
```

MLflow with the framework-specific autologger:

```Python
mlflow.xgboost.autolog()
```

The features above not only stores the model parameters for ease of use. It also stores other files inside the model folder, artifact folder. But still can be specified.


## Saving Models

We also can save the model in the same way to log others things above. 

mlflow:

```Python
mlflow.log_model(" your model ", artifact_path="models_mlflow")
```

or

mlflow with framework:

```Python
mlflow.<framework-name>.log_model(" your model ", artifact_path="models_mlflow")
```

 ## Saving Artifact
 
 In some experiment or training we have some artifacts to use in the model. from my notebook, It is the `DictVectorizer` that I have used to transform the data. I have to save this artifact with the model to use in the inference step in the testing.
 
 ```Python
 mlflow.log_artifact("vertorizer.pkl", artifact_path="artifact") # the path for store artifact
 ```
 
 ## Loading Models
 
 In case that to use the model to make prediction with multiple ways depending on what you need or your scenario
 MLflow let you load the model with many ways:
 * Spark UDF for Spark Dataframes
 * PyFuncModel structure for Pandas Dataframes, Numpy Array, Scipy Sparse Array
 * Model's framework e.g. XGBoost or Tensorflow or Pytorch

```Python
run_id = " your run ID "
logged_model = f"runs:/{run_id}/models"

xgboost_model = mlflow.xgboost.load_model(logged_model)
```

The XGBoost object that behaves like any XGBoost model. So we can make prediction as normal.
