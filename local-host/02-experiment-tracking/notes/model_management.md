# MODEL MANAGEMENT
---

![MODEL MANAGEMENT](https://raw.githubusercontent.com/surawut-jirasaktavee/course-mlops-zoomcamp/main/local-host/02-experiment-tracking/images/MLOps_cycle.webp)

ref: [neptune.ai/blog](https://neptune.ai/blog/ml-experiment-tracking)

Experiment tracking focuses on the iterative model development phase will starts when models go to production:
* streamlines moving models from experimentation to production
* helps with model versioning
* organizes model artifacts in an ML model registry
* enables rolling back to an old model version if the new one seems to be going crazy

But not every model gets deployed. The Experiment tracking is still useful. Because this phase will save all the medatada about every experiment you run ensures that you will be ready when this magical moment happens.

The way that needs to use to avoid the error-prone of any mistakes in the experiment that are not small or you don't work alone is to don't use `Spreadsheets or files + naming conventions`. Using modern experiment trackings tools like `MLflow`, `Neptune.ai` or `Wandb` can help you easily work between your experiment.
