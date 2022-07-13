# MLFLOW_BENEFIT, LIMITATION AND ALTERNATIVES
---

## Remote tracking server

**Deploying the remote tracking server on the cloud provider give some benefits**:

- Share experiments with other data scientist
- Collaborate with others to build and deploy models
- Give more visibility of the data science efforts

**Issues with running a remote MLflow server**:

- Security
    - Restrict access to the server (e.g. VPN)
- Scalability
    - Check [Deploy MLflow to AWS Fargate](https://github.com/aws-samples/amazon-sagemaker-mlflow-fargate)
    - Check [MLflow at Company Scale by Jean-Denis Lesage](https://databricks.com/session_eu20/mlflow-at-company-scale)
- Isolation
    - Define standard for naming experiments, model and a set of default tags
    - Restrict access to artifacts (e.g. use s3 buckets living in different AWS accounts)

### MLflow limitations 

- **Authentication & Users**: The open source version of MLflow doesn't provide any sort of authentication
- **Data versioning**: to ensure full reproducibility we need to version the data used to train the model. MLflow doesn't provide a built-in solution for that but there are a few ways to deal with this limitation.
- **Model/Data Monitoring & Alerting**: this is outside of the scope of MLflow and currently there are more suitable tools for doing this.

## MLflow alternatives

- [Neptune](https://neptune.ai/)
- [Comet](https://www.comet.com/site/)
- [Weight & Bias](https://wandb.ai/site)
- [Many more](https://neptune.ai/blog/best-ml-experiment-tracking-tools)