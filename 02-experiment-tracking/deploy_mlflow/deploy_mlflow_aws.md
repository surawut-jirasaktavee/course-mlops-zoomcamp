# Deploy MLflow on AWS cloud provider
---

## EC2 Instance

## Create EC2 instnace for Remote Tracking Server

1. Goto the **EC2** page and `Launch instance` 

![Launch instance](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/lunch_instance.png)

2. Set instance **Name and tags**

![Name and tags](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/setup_instance_name.png)

3. Select the **OS** for `EC2 instance with 64-bit Architecture`

![Select OS](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/select_os_instance.png)

4. Select the **Instance type** by your needed.

![Instance Type](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/089e105b95433422ba4d1881c45dcf105635d459/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/select_instance_type.png)

5. Configure storage size (Free tier eligible customers can get up to 30GB)

![Configure storage](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/configure_storage_size.png)

6. Create new key pair in case that you want to connect with `SSH`

![Create new key pair](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/create_key_pair_for_ssh.png)

![Create key pair name](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/set_key_pair_name.png)

**Note**: Once create key pair you need to save or download you key pair file because just only one time that you can find it when you created it.

7. Leave the rest as default and `Lunch instance` in the **Summary** pane

![Summary pane](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/create_instance.png)

8. Check your instance from the **Instance info** page

![Instance info](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/check_instance_from_ui.png)

## Create security rule 

1. After finished to create the `EC2 Instance` under the **Network & Security** tab go to `Security Groups`

![Security Group](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/security_group.png)

2. On the **Security Group** page click on `Security group ID` that is the same of your `Security group name` of your `EC2 instance`

![Security group ID](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/security_group.png)

**Note**: you can check `Security group name` of your instance by go to your instance and under the `security` tab.

![Check security group](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/check_security_group.png)

3. Add new rule to **inbound rule** of your instance
I have customize to `Custom TCP` with `TCP Protocol` and `Port range 5000` and `Anywhere IPv4` 

4. Save new rule to use with your instance

![Save new rule](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/save_new_rule.png)

![Add new rule](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/new_rule.png)

## Create RDS Postgresql database for Backend Store

## Create S3 Bucket for Artifacts Store
