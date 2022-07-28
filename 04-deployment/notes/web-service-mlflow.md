# Deploy the Machine Learning model with MLflow using Flask and Docker
---

MLflow is the `model management` tools that help you to manage Machine Learning project. So you can manage:

- Model life cycle
- Model versioning
- Model logging
- etc

I already use the `MLflow` with **Remote Tracking Server** and **Backend store** with `AWS RDS in **Postgresql**`, and use **Artifacts store** in `AWS S3`.

Regardless from this module. I have used the Machine Learning model with MLflow and store the **Artifacts** in `AWS S3` I will retrieve the model from the `AWS S3` and then deploy the model using `Flask` and `Docker`.

The main topic of this module is the way we stored, retrieve, and deploy the model.

To store the model with `MLflow` in `AWS S3` artifacts store. You have to set the `tracking uri` with mlflow:

**For example**:

```Python
import mlflow

MLFLOW_TRACKING_URI = "<Publick IP or DNS of remote server>"
mlflow.set_tracking_uri(f'http://{MLFLOW_TRACKING_URI}:5000')
mlflow.set_experiment("experiment_name") # In case that is the first time of the experiment
```

Then you can train the model and log the model, parameters, and etc.

To retrieve the model from `Artifacts store` (in this case is AWS S3):

```Python
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "<Publick IP or DNS of remote server>"
RUN_ID = "RUN_ID" # Check in the Mlflow model page

client = MlflowClient(tracking_uri=f'http://{MLFLOW_TRACKING_URI}:5000')

# s3://mlflow-artifacts-prem/1/3fc5b94495d54d978b8dcb5094cccdcf/artifacts/model
model = client.download_artifacts(run_id=RUN_ID, path='/model/model.pkl')
```

The MLflow have many way to load the model. [Other ways](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model)

In case that you want to use **FULL PATH** in S3 and if you use the particular profile you have to set the `environment variable` then the boto3 will look to your profile from the `~/.aws/config`

In the terminal:

```zsh
export AWS_PROFILE = "your profile"
export AWS_REGION = "your user region"
export AWS_ACCESS_KEY = "your access key"
export AWS_SECRET_ACCESS_KEY = "your secret key"
```

In code:

```Python
import os

os.getenv('AWS_PROFILE')

RUN_ID = os.getenv('RUN_ID') # if you use env variable with run id you can do it.
logged_model = f's3://path/{RUN_ID}/artifacts/model'
model = mlflow.pyfunc.load_model(logged_model)
```

**DON'T FORGE TO CREATE THE PIPFILE TO USE FOR THE DEPLOYMENT**

Inside the working directory:

```Python
pipenv install packages
```

To activate the Pip Shell:

```Python
pipenv shell
```

Check the deployment step from [web-service](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/notes/web-service.md)