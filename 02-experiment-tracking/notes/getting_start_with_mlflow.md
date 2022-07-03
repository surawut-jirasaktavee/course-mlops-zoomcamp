# GETTING START WITH MLFLOW
---

## [Installing MLflow](https://mlflow.org/docs/latest/quickstart.html#installing-mlflow)

pip:

```Python
pip install mlflow
```

or

```Python
pip3 install mlflow
```

conda:

```conda
conda install -c conda-forge mlflow
```

## [Viewing the Tracking UI](https://mlflow.org/docs/latest/quickstart.html#viewing-the-tracking-ui)

```Python
mlflow ui
```

By default, The tracking API writes data into files into a local ~/mlruns directory.

To run the MLflow UI locally run the command:

```zsh
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
The backend storage is essential to access the features of MLflow. This command use a SQLite backend with the file `mlflow.db` in the current running repo.

For an artifact, we can also add an artifact root directory where we store the artifacts for runs by adding a `--default-artifact-root` parameter:

```zsh
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
