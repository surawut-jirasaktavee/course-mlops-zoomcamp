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

But the `MLflow` **dont' put your model to the production or move to other stages**. you need to complement the model registry with some `CI/CD` tools.
