# Deployment Prefect Flow
---

I have create the remote server for the MLflow to the model management and model registry with:
 - AWS S3 Bucket for the artifacts store.
 - AWS RDS Postgresql for the backend-ui store.
 - AWS EC2 to use in case fo remote tracking server.

I also use Prefect to orcestrate the machine leanring pipeline. I have use Prefect remote server as well and store Prefect artifact in the S3 bucket.

To create the Prefect artifact store I have create the function to create the S3 object with Prefect API:

```python
from prefect.filesystems import S3

bucket_name = "mlflow-artifacts-prem"
os.environ["AWS_PROFILE"] = "MLOps-dev"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

def get_storage(s3_bucket: str, 
                AWS_ACCESS_KEY_ID: str,
                AWS_SECRET_ACCESS_KEY: str):

    block = S3(bucket_path=s3_bucket, 
            aws_access_key_id=AWS_ACCESS_KEY_ID, 
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    return block
    
block = get_storage(bucket_name, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
```

For the full script see: [prefect_deploy.py](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/03-orchestration/prefect_deploy.py)

> In order to deploy the Prefect in the Prefect > 2.0.1(Current Patch)
> Run the command 

I have specific:
- the command `prefect deployment build`
- the root location to build the deployment with the name of the file to deploy and the function: `./prefect_deploy.py:main`
- the name of the deployment (optional if not specify then will be generate)
- tag (optional)
- infra type [process, docker, kubetenes]
- storage-block (type of the storage/storage name)

```bash
prefect deployment build ./prefect_deploy.py:main \
--name xgboost_optimization \
--tag dev \
--infra process \
--storage-block s3/mlflow-artifacts-prem
```

If the build command can proces successfully then we will get 2 files:
1. main.menifest.json
2. deployment.yaml

Then we have to apply the deployment script with the command:

```bash
prefect deployment apply
```

The command will use the files from previos command to create the deployment then we need to create the agent and create the flow and task to run as schedule or anything else.

To create the `Prefect agent` ...
