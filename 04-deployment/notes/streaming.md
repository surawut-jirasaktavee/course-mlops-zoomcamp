# Machine Learning for Streaming with AWS

* Scenario
* Creating the role
* Create a Lambda function, test it
* Create a Kinesis stream
* Connect the function to the stream
* Send the records

## Scenario

This time I already have the model and can predict the `ride duration` in this case the taxi trip. So I think this model should be deploy as the streaming because the customer of the taxi want to know the ride duration of the trip. So let's deploy it as stream!

First of all. I will deploy the model with `AWS services`.

## Creating the role

In `AWS` once you have created the account you will given the accoutn that you created as a root user. this user can do anything on `AWS`. but this root user is so convenient and easy for the others people that get your credential can do anythings. check [AWS IAM](https://aws.amazon.com/iam/)

**THE BEST PRACTICE** is you should create the user e.g. `Administrator`, `MLOps`. and assign the role & policy to that user as their privilege by their role. I already create my `MLOps-dev` user to work in this project. Then I will assign the role that needed to work with streaming model.

For create the role you should be the permission to do this as well. ensure that you have it.

1. Go to the name of your account at the top right and click to show the dropdown.
2. Security credentials >> Roles.
3. Create role.

    ![create_lambda_role](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/create_lambda_role.png)

4. In **Trusted Entity tpye** select `AWS Service`.
5. In the **User case** select `Lambda`.

    ![lambda_role_config](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/lambda_role_config.png)

6. Create the role for **Lambda & Kinesis**. In the search bar search the following keyword:
    * AWSLambdaKinesisExecutionRole

    ![lambda_add_execution_role](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/lambda_add_execution_role.png)

7. Set the role name and discription of the role.

    ![lambda_set_role_name](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/lambda_set_role_name.png)

8. Back to the **Role** page and select `Add permission` and then add the role that created.

    ![select the new role](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/select_the_new_role.png)

    ![Add permission](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/add_permission.png)

    ![add_lambda_permission](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/add_lambda_permission.png)

9. Now you finish to add the role to work with stream by `Lambda` & `Kinesis` services.

## Create a Lambda function, test it

1. Go to `Lambda` page or search in the search bar on the top of page.

    ![Lambda](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/lambda.png)

2. Click on `Create funtion` botton.

    ![Create function](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/lambda_create_func.png)

3. Select the `Author from scratch`  and set the name of Lambda.

    ![lambda_function_name](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/lambda_function_name.png)

4. Select the languages that you want lambda to support and select the existing permission from the role that you have created.

    ![lambda_select_lang](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/lambda_select_lang.png)

    ![lambda_permission](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/lambda_permission.png)

5. Now you finish to create `Lambda function`. Go around with the function you want and test it.

    ![lambda_finish](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/lambda_finished.png)

## Create a Kinesis stream

1. Go to `Kinesis` page or search on the search bar.

    ![Kinesis](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/kinesis.png)

2. Click on `Kinesis create data stream`.

    ![kinesis_create_data_stream](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/kinesis_create_data_stream.png)

3. Set your Kinesis name and set others capacity of your Kinesis. Then create it.

    ![kinesis_data_stream_setup](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/kinesis_data_stream_setup.png)

4. Now you have `Lambda` and `Kinesis`.

5. Let's add `Trigger` to the Lambda function.

    ![add_kinesis_consumer](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/add_kinesis_consumer.png)

    ![select_the_event](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/select_the_event.png)

    ![lambda_kinesis](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/lambda_kinesis.png)

* [Tutorial: Using Amazon Lambda with Amazon Kinesis](https://docs.amazonaws.cn/en_us/lambda/latest/dg/with-kinesis-example.html)

For this stream model. I have store the artifacts of the model in the `AWS S3`. According to this way I have to work with the `boto3` to use the client to connect with `Kinesis` and use `mlflow` to download the model in the bucket.

In ordered to download the model from **S3**. We have to add new policy to let us list the data from **S3** as well.

1. Go to the `IAM` page and `Roles` select the role that created and `Add permission` >> `Attach policy` >> `Create policy`.
    * Choose the sevice: `S3`

    ![create_s3_policy](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/create_s3_policy_for_docker.png)

2. Add **Access level**:
    * List any
    * Read any

    ![add_s3_access_level](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/add_s3_access_level.png)

3. Add ARN(Amazon Resource Name) = The bucket name.

    ![add_s3_bucket_name](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/add_s3_bucket_name.png)

4. Add Object
    * Bucket name
    * Object in the backet (select any or "*" then you can list and read every object in the bucket if you needed)

    ![add_s3_access_n_object](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/add_s3_bucket_n_object.png)

5. Back to the `Add permission` page and add the S3 policy to the role.

    ![add_s3_policy](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/add_s3_policy.png)

According to make the prediction after we get the result. We have to send out the prediction result to stream to somewhere else. we need to add `Put Record` or `Put Records` permission to the Kinesis. That will make Kinesis can write the result out.

1. Follow the step from the previous to go to the `Create Policy` page.
2. Select the `Kinesis` service.

    ![select_kinesis_service](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/select_kinesis_service.png)

3. Set the `Access level`.
    * Put Record
    * Put Records

    ![set_put_record_access](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/set_put_record_access.png)

4. Set the ARN.
    * Region
    * Account ID
    * Stream name

    ![set_the_arn_put_record](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/set_the_arn_put_record.png)

    ![kineiss_add_permission_finish](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/kinesis_add_permission_finish.png)

5. Back to the role page and add this policy to the role.

## Code snippets

In order to use `Kinesis client` to send to predicton result out:

```python
import boto3

kinesis_client = boto3.client('kinesis')

response = kinesis_client.put_record(
                StreamName=PREDICTIONS_STREAM_NAME,
                Data=json.dumps(prediction_event),
                PartitionKey=str(ride_id),
            ) 
```

The way I have to download the model from the `MLflow Artifact store` from the **S3**.

```python
import os
import mlflow

RUN_ID = os.getenv('RUN_ID')
os.environ["AWS_PROFILE"] = "YOUR-AWS-PROFILE"
logged_model = f's3://mlflow-artifacts-prem/1/{RUN_ID}/artifacts/model'
model = mlflow.pyfunc.load_model(logged_model)
```

To test the script in `Lambda` you can use test tab on AWS.

check the [lambda_function.py](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/lambda_function.py)

check the [test.py](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/test.py)

## Connect the function to the stream

To connect the function to the stream. I have deploy with docker container.

1. Create virtual environment to use with docker. Specify the version of each package of your environment.

    ```bash
    pipenv install boto3 mlflow scikit-learn==1.1.1 --python=3.9
    ```

    To activate virtual environment run the command:

    ```bash
    pipenv shell
    ```

2. To create the Dockerfile I used `lambda/python` image from the `AWS Gallery`

    ```Dockerfile
    FROM public.ecr.aws/lambda/python:3.9

    RUN pip install -U pip
    RUN pip install pipenv

    COPY [ "Pipfile", "Pipfile.lock", "./" ]

    RUN pipenv install --system --deploy

    COPY [ "lambda_function.py", "./" ]

    CMD [ "lambda_function.lambda_handler" ]
    ```

    To visit other gallery in Amazon ECR Public Gallery see: [AWS ECR Gallery](https://gallery.ecr.aws/lambda/python)

    For the Lamba/Python image see: [lambda/python](https://gallery.ecr.aws/lambda/python)

3. To build Docker image run the command:

    ```bash
    docker build --platform=arm64 -f Dockerfile -t stream-model-duration:v1 .
    ```

4. In case that use AWS CLI, you may need to set the env variables:

    ```bash
    export AWS_PROFILE="YOUR_AWS_PROFILE"
    export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID" 
    export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY" 
    export AWS_DEFAULT_REGION="YOUR_AWS_DEFAULT_REGION" 
    ```

    Or you can set environment variable on the `Lambda` in AWS as well.

5. To run the docker image that created run the command:

    I have to set the environment variables in to the docker as well bacause I have created the account to work in this zoomcamp. And I have mounted my `AWS Credentials` file in my local and my function that I will deploy.

    ```bash
    docker run -it --rm -d \
        -p 8080:8080 \
        -e PREDICTIONS_STREAM_NAME="ride_predictions" \
        -e RUN_ID="3fc5b94495d54d978b8dcb5094cccdcf" \
        -e TEST_RUN="True" \
        -e AWS_PROFILE="${AWS_PROFILE}" \
        -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
        -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
        -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
        -v /Users/premsurawut/.aws:/root/.aws \
        stream-model-duration:v1
    ```

6. To creating an ECR repo on AWS:

    ```bash
    aws ecr create-repository --repository-name duration-model
    ```

    **Note**:You will get some imformation about the repository url to use in the next step.

7. Publishing Docker images:

    Logging in...

    ```bash
    aws ecr get-login-password \
    --region us-west-1 \
    | docker login \
    --username AWS \
    --password-stdin "551011018709.dkr.ecr.us-west-1.amazonaws.com/duration-model"
    ```

    See more about [Amazon ECR public registries](https://docs.aws.amazon.com/AmazonECR/latest/public/public-registries.html)

8. Pushing the docker image:

```bash
REMOTE_URI="551011018709.dkr.ecr.us-west-1.amazonaws.com/duration-model"
REMOTE_TAG="v1"
REMOTE_IMAGE=${REMOTE_URI}:${REMOTE_TAG}

LOCAL_IMAGE="stream-model-duration:v1"
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE}
docker push ${REMOTE_IMAGE}
```

## Send the record

### Sending data

TEST LOCALLY

To sending this record:

* Set the name of the `KINESIS_STREAM_INPUT`.
* Use the `AWS SDK` to send the `put-record` to the **Lambda Function**.
  * Define the `stream-name.
  * Define partition-key of your Kinesis.
  * Define the sample data.

Run the command on the terminal with `AWS CLI`:

```bash
KINESIS_STREAM_INPUT="ride_events"
```

```bash
aws kinesis put-record \
    --stream-name ${KINESIS_STREAM_INPUT} \
    --partition-key 1 \
    --cli-binary-format raw-in-base64-out \
    --data '{
        "ride": {
            "PULocationID": 90,
            "DOLocationID": 285,
            "trip_distance": 3.22
        },
        "ride_id": 144
    }'
```

After run the above command your will get the `shard_id` and `sequenceNumber` to print out to the terminal.
I have to use this number in the next step.

### Test event

#### Reading data from the stream

To read the data from the stream locally run to command:
    - Define the name of the `KINESIS_STREAM_OUTPUT`.
    - Define the SHARD with the `shard_id` from the previous command.
    - Define the shard-terator-type with the `sequenceNumber` from the previous command.
    - Define the query type.
    - Define the `starting-sequence-number` to start to run the command with the number that you got from previous command.
    - Store the command to **SHARD_ITERATOR** variable.
    - Run the `get-records` command of the Kinesis and store the result to the variable.
    - Print the result to reading data from the stream.
    - This is the data from the previos command that send to the **Lambda function**.

```bash
KINESIS_STREAM_OUTPUT='ride_events'
SHARD='shardId-000000000000'

SHARD_ITERATOR=$(aws kinesis \
    get-shard-iterator \
        --shard-id ${SHARD} \
        --shard-iterator-type AT_SEQUENCE_NUMBER \
        --stream-name ${KINESIS_STREAM_OUTPUT} \
        --query 'ShardIterator' \
        --starting-sequence-number 49631818369979129679861815788598519962499637577139617794
)

RESULT=$(aws kinesis get-records --shard-iterator $SHARD_ITERATOR)

echo ${RESULT} | jq -r '.Records[0].Data' | base64 --decode | jq

```

#### Running the test

In order to test locally I have to set the environment variable to use as the parameter and send to the `lambda_function.py` as following.

```Bash
export PREDICTIONS_STREAM_NAME="ride_predictions"
export RUN_ID="3fc5b94495d54d978b8dcb5094cccdcf"
export TEST_RUN="True"

python test.py
```

Check [test.py](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/test.py)

The JSON data below is the data that I have put to the Lambda from a few previous command. You can get the record as the information like this from the `CloudWatch` UI.  I will use this information in the `test script`.

```json
{
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49631818369979129679861815788598519962499637577139617794",
                "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogOTAsCiAgICAgICAgICAgICJET0xvY2F0aW9uSUQiOiAyODUsCiAgICAgICAgICAgICJ0cmlwX2Rpc3RhbmNlIjogMy4yMgogICAgICAgIH0sCiAgICAgICAgInJpZGVfaWQiOiAxNDQKICAgIH0=",
                "approximateArrivalTimestamp": 1659013681.735
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49631818369979129679861815788598519962499637577139617794",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::551011018709:role/lambda-kinesis-role",
            "awsRegion": "us-west-1",
            "eventSourceARN": "arn:aws:kinesis:us-west-1:551011018709:stream/ride_events"
        }
    ]
}
```

![cloudwatch](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/cloudwatch.png)

![logstream](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/logstream.png)

To test the ECR I have use the `test_docker.py` to test it. The script are the same except the **URL like** that need to use for test the ECR image.

* [URL for testing:](http://localhost:8080/2015-03-31/functions/function/invocations)
* [Reference:](https://docs.aws.amazon.com/lambda/latest/dg/images-test.html)

#### TSET RESULT

![test_result](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/04-deployment/streaming/images/test_results.png)
