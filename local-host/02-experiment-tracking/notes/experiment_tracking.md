# EXPERIMENT TRACKING
---

## Important concept

* **ML Experiment**: the process of building an ML model
* **Experiment run**: each trial in an ML experiment
* **Run artifact**: any file that is associated with an ML run
* **Experiment metadata**

What's experiment tracking?

Experiment tracking is the process of keeping track of all the **relevant infomation** from an `ML experiment`, which includes:
* Source code
* Environment
* Data
* Model
* Hyperparameters
* Metrics
* etc.

These 3 main reason are so important for experiment tracking:
* Reproducibility
* Organization
* Optimization

For example, In the Kaggle competition you may need to `reproduction` some of the notebooks of Grandmaster to check some things or for a base model and then you may need some `organization` to help you while you do the experiment or `optimization` the model and maybe tracking experiment with spreadsheets that not enough because you can make the `error prone` or maybe you have to collaborate with your team. So it comes to a lot of formats that `don't have standards`. Some tools like, for example `MLflow` can help you to solve these problems and increase `visibility` and `collaboration` with your team.

[MLflow](https://mlflow.org/) an open source platform for the machine learning lifecycle. The Python package that can be installed with `pip` and contains for main modules:
* [Tracking](https://mlflow.org/docs/latest/tracking.html)
* [Models](https://mlflow.org/docs/latest/models.html)
* [Model Registry](https://mlflow.org/docs/latest/model-registry.html)
* [Projects](https://mlflow.org/docs/latest/projects.html)
* [MLflow docs](https://mlflow.org/docs/latest/index.html)



