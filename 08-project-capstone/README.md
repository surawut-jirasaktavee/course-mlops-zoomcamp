# Course MLOps-Zoomcamp Final Project Capstone

## Project name

Telecom customer churn prediction

## Project evaluation criteria:
https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/07-project

## Project description

This is the final project for the course `mlops-zoomcamp` from [DataTalksClub](https://github.com/DataTalksClub/mlops-zoomcamp).

**Project pain point**:
Churn prediction means detecting which customers are likely to leave or cancel a service subscription. It is a critical prediction for many businesses because acquiring new clients often costs more than retaining existing ones

Since I am the Data Engineer in the Telecommunication industry and I have some interested about the Data Science. I have decided to combine my interesting and my current industry and do some project with the course `mlops-zoomcamp` for the final project.

The project provides the online service for the prediction of customer who will has the possibility to **Churn** from the company. 

The project's focus to make a **Production service** with experiment tracking using **MLflow**, pipeline automation using **Prefect**, observabilitty using **Evidently**, **Prometheus**, and **Grafana**.

## Dataset

The dataset has been taken from [Kaggle](https://www.kaggle.com/competitions/customer-churn-prediction-2020/overview). 

See data dictionary: [Data_Dictionary](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/08-project-capstone/dataset/README.md)


## Project preparations

This project is implemented on `Ubuntu 22.04` on AWS as below:

Platform details: Linux/UNIX

AMI name: ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-20220609

Reproducibility are based on specific configuration above and already proved with `Mac M1 Air`

This project repository contains 2 folders.

1. dataset folder contains train and test datasets.
2. code folder contains the main source code with configurations file includes.

  - Dockerfile
  - docker-compose.yml
  - MLflow database
  - Prefect database
  - Prediction services
  - CI/CD pipeline
  - Integration test



## Project overview

The project start by training session to train the model with "`model_training.py`" and promoting the model to the model registry with "`model_registry.py`" and store the model artifacts and saving the model with "`pickle`" file. **MLflow** being used for model experiment tracking, registry, store artifacts by save to the "`customer_chrun.db`" 

In order to make the machine learning pipeline then **Prefect** come to work on the project. Prefect has been used as a workflow orchestrator in the project by deploy the project create the `task` and `flow` then schedule the pipeline to run every time.

Now after the model training session and deployment process. The project still not ready and need more task to work. This time the model should be use and serving the expect performance from the model as application that will use by the users this application will implement using **Flask**. 

Apart from this the observability for the service is the task being implement to ensures the opearation team can observability by a combination of **Grafana**, **Prometheus**, and **Evidently** with these services will provide real-time of model performance and data drift from the production. So the operation can handle the problems immediately or requests others team to help and discuss to make the decission together by time.

The demo below will simulate the traffic of the model service in real-time prediction pipeline by the `customer_data.py` script. It will send the input features from the [dataset](https://github.com/DataTalksClub/mlops-zoomcamp/08-project-capstone/dataset/churn_test_dataset.csv) in every a few seconds and send the response back to the **Data Mart** (Assumed) in the "`./code/churn_report/customer_churn_result.csv`" with **customer ID** and **Churn result**.


## Demo

This demo shows the main functionality of the project and will follow by the project instructions below.


![]()

## Project instruction

### 1. Project repository

```bash
git clone https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp.git
```

Clone the project to the local server.

### 2. Work directory

```bash
cd ./course-mlops-zoomcamp/08-project-capstone/code
```

Move to the `./course-mlops-zoomcamp/08-project-capstone/code` from the current directory.

### 3. Build the services up

```bash
docker-compose up --build
```

This command will run the docker-compose to build up the services and dependencies for the services.

**NOTE**: Add `-d` to run in the detach mode


### 4. Environment preparation

```bash
pipenv shell
```

This command will install all required dependencies and activtte the environment from this requisition of this project.

### 5. MLflow preparation

```bash
mlflow ui -h 0.0.0.0 -p 5050 --backend-store-uri sqlite:///customer_churn.db --default-artifact-root ./mlruns
```

This command will build the mlflow service to work in this project e.g. `database` and `MLflow UI`.

**link**: http://localhost:5050/

### 6. Prefect preparation

```bash
prefect config set PREFECT_API_URL="http://0.0.0.0:4200/api" # local server
prefect orion start --host 0.0.0.0
```

The command above will set the `PREFECT API URL` at localhost with port 4200 and start `prefect orion` 

**link**: http://0.0.0.0:4200/

**NOTE**: for use prefect as remote server need to set with the command:

```bash
prefect config set PREFECT_ORION_UI_API_URL="http://<external ip>:4200/api" # Remote server
```

### 7. MLflow Model training and monitoring

```bash
python model_training.py
```

Run python script `model_training.py` to start training the model. For this project use `LightGBMClassifier` to classification the data between customer who will **Churn** or  **Not Churn**.

In order to inspect the training process or details apart from command line. Open the `MLflow UI` or `Prefect UI` instead.

The model will transition to the `Staging` in the `MLflow UI` after finish training session.

### 8. MLflow Model registry and artifacts

```bash
python model_registry.py
```

Run `model_registry.py` to retrieve the data of the model from the `MLflow database` and compare current model in the production with the new model. If the new model is better then transition this new model to the production and give the archive stage to the production model.

### 9. Prefect deployment

```bash
prefect deployment create model_deployment.py
```

The command above will create the deployment from `model_deployment.py` and return the **deployment ID** in the terminal.

### 10. Prefect work-queue

```bash
prefect work-queue create -d <deployment ID> <work-queue name>
```

In order to see the deployment ID run the command:

```bash
prefect deployment ls
```

The work-queue should be created.

### 11. Prefect agent

```bash
prefect agent start <work-queue ID>
```

After created the work-queue. now the `Prefect agent` can work on the deployment to help the pipeline run in the specify time by schedule the deployment

In order to see the work-queue ID run the command:

```bash
prefect work-queue ls
```

### 12. Model prediction

Now the model ready to use after training and deploy to the production and the services should be ready to serve the prediction result of each customer who will **Churn** or **Not Churn** from our company.

In the current directory `(./course-mlops-zoomcamp/08-project-capstone/code)` run the command below to send the data to the model service and get the results.

```bash
python customer_data.py
```

this command will read the data from `./evidently_service/datasets/churn_test_dataset.csv` and send it to the `Flask` application. And the model service will send the outputs the prediction as **Churn** or **Not Churn** and write the `customer ID` and `churn result` to `./churn_report` to serve this result to the analyst team to work on this result and find the good solutions.

### 13. Model & Data monitoring

In order to inspect the `data drift` or `model drift` see the **Grafana dashboard** to monitor then check if any data show abnormal or model performance become to bad.


## Future improvement

This project still need more improvement to give more abilities for work with. 

1. Add the alert when the data drift or model performance is down.

2. When the alert occurs send some triggers. For example when the model performance is down show the alert to the operation team and send the trigger to the prefect to start the model deployment to re-train the model to get better model.

3. Combine both two above with CI/CD process this will become to CI/CD/CT pipeline and deploy to the production automatically.

4. Add others model to training session and select the best model for the project.

5. Add model optimization such as `hyperopt` to tune hyperparameters.

6. Add IaC to manage and deploy the infrastructure and services.


## Tech Stack

- Flask
- MLflow
- Prefect
- Grafana
- Prometheus
- Evidently
- Mongodb
- Docker
