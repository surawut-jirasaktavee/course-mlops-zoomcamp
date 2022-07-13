# Deploy MLflow on AWS cloud provider
---

## AWS EC2 Instance

### Create EC2 instnace for Remote Tracking Server

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

![Instance info](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/check_instance_from_ui_edit.png)

### Create security rule of EC2 instance

1. After finished to create the `EC2 Instance` under the **Network & Security** tab go to `Security Groups`
2. On the **Security Group** page click on `Security group ID` that is the same of your `Security group name` of your `EC2 instance`

![Security group ID](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/security_group.png)

**Note**: you can check `Security group name` of your instance by go to your instance and under the `security` tab.

![Check security group](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/check_security_group_edit.png)

3. Add new rule to **inbound rule** of your instance
I have customize to `Custom TCP` with `TCP Protocol` and `Port range 5000` and `Anywhere IPv4` 

4. Save new rule to use with your instance

![Save new rule](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/save_new_rule.png)

![Add new rule](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/new_rule.png)

### Connect to EC2 Instance with SSH

**Follow thses step to connect with your instance by SSH**

![Connect to instance](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/connect_to_instance_edit.png)

1. Open your local terminal
2. Locate your private key file that you have create in the EC2 process. The file extension is `<your key pair name>.pem` in my case is `prem-mlops-zoomcamp-pem`
3. Run this command, to ensure your key is not publicly viewable.

```Zsh
chmod 400 prem-mlops-zoomcamp-pem
```

4. Connect to your instance using its Public DNS from the EC2 instance page
For example my case: 

![Public DNS](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_ec2_instance/public_dns_edit.png)

```Zsh
ssh -i "prem-mlops-zoomcamp.pem" ubuntu@ec2-<your_public_ip>.us-west-1.compute.amazonaws.com
```

**Optional**: 

To avoid typing the full `SSH` command you and set the config and then you can just type `SSH` command with your instance ID(shortly)
For example:

```Zsh
ssh <your instance ID>
```

follow these code snippet below:

```Zsh
nano ~/path/to/.ssh/config
```

open the config file and set your credential in the file.

```Zsh
Host mlops-zoomcamp-prem
    Hostname <your_public_ip>
    User ubuntu
    IdentityFile ~/path/to/save/<credentials>.pem
    StrictHostKeyChecking no
```

To execute the config file.

```zsh
source ~/path/to/.ssh/config
```


## AWS RDS PostgreSQL Database

Go to Amazon RDS page to create the Databases

![AWS RDS](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/rds_postgres/go_to_rds.png)

### Create RDS Postgresql database for Backend Store

1. On Amazon RDS page select `Create database` on **Amazon Aurora** pane

![Crate db](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/rds_postgres/create_db.png)

2. Under **Create database** & **Engine option** pane, Choose a database creation method, In my case I have selected `Standard create` and selected `PostgreSQL` database

![Create PostgreSQL](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/rds_postgres/select_postgres.png)

3. Under the **Setting** pane put your DB cluster identifier and your master username and password, In my case I selected the `Auto generate a password` 

![DB information](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/rds_postgres/setting_postgres.png)

**Note**: If you select to `Auto generate a password` once you click to create the database in to UI on the top blue pane pop-up you have to download the generated password from the **AWS** that is the one time generate for you.

4. Under the **Instance configuration** select the DB instance class as you want

![DB instance class](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/rds_postgres/select_db_instance_class.png)

5. Under the **Connectivity** pane on the `Public access` select `No` for allow only Amazon EC2 instances and devices inside the VPC can connect to the database.

![Public access](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/rds_postgres/public_access_no.png)

And open the **Additional configuration** on bottom of **Connectivity** pane to configure the Database port to `5432`.

![db port](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/rds_postgres/db_port.png)


6. Don't forget to Initial database name. Open **Additional configuration** pane and put your database name

![Initial db name](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/rds_postgres/initial_db_name.png)

### Create Security rule for PostgreSQl Database

1. Under the **Connectiviti & security** pane go to VPC security groups

![VPC security groups](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/rds_postgres/inside_db_tab.png)

2. Goto `Inbound rules` tab. Click to `Edit inbound rules`

![Inbound rule tab](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/rds_postgres/inbound_rule_tab.png)

3. Add rule for `PostgreSQL type` with `TCP Protocol`, `Port range 5000` and select the **Security Groups** with the same of your `EC2 Instance` 

![New rule](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/rds_postgres/add_security_rule.png)

4. Save the rule and check Database status.

## AWS S3 Bucket

Search for `S3` and open the **S3 features**

![S3](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/s3_bucket/goto_S3.png)

### Create S3 Bucket for Artifacts Store

1. Click to `Create bucket` 

![Create bucket](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/s3_bucket/create_s3_bucket.png)

2. Under **General configuration** pane. set up your `Backet name`

![Set bucket name](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/s3_bucket/setup_bucket_name.png)

3. leave the rest as default and create the bucket, Then you will see your bucket on the `Bucket` page

![Finished create bucket](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/s3_bucket/setup_bucket_name.png)

**Additional

## AWS Command line interface

### AWS CLI installation instructions

1. Download the `AWS CLI` for the [link](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

2. Open installation file and choose the following ways to the **AWS CLI**

    - **For all users on the computer (requires sudo)**
        - You can install to any folder, or choose the recommended default folder of `/usr/local/aws-cli`.
        - The installer automatically creates a symlink at `/usr/local/bin/aws` that links to the main program in the installation folder you chose.

    - **For only the current user (doesn't require sudo)**
        - You can install to any folder to which you have write permission.
        - Due to standard user permissions, after the installer finishes, you must manually create a symlink file in your `$PATH` that points to the aws and aws_completer programs by using the following commands at the command prompt. If your $PATH includes a folder you can write to, you can run the following command without sudo if you specify that folder as the target's path. If you don't have a writable folder in your `$PATH`, you must use sudo in the commands to get permissions to write to the specified target folder. The default location for a symlink is `/usr/local/bin/`.

        ```Zsh
        sudo ln -s /folder/installed/aws-cli/aws /usr/local/bin/aws
        sudo ln -s /folder/installed/aws-cli/aws_completer /usr/local/bin/aws_completer
        ```

        > Note
        > You can view debug logs for the installation by pressing Cmd+L anywhere in the installer. This opens a log pane that enables you to filter and save the log. The log file is also automatically saved to /var/log/install.log.
        
3. To verify that the shell can find and run the aws command in your $PATH, use the following commands. And then try to interface with my s3 and found some error as below, That told you to configure your **AWS configuration** first

```Zsh
which aws
```

```Zsh
aws --version
```

![Verify CLI](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_cli/aws_cli_install.png)

4. You can configure **AWS CLI** with following command

```Zsh
aws configure
```

![AWS Configure](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_cli/configure_aws_edit.png)

Then you will prompt to put your **AWS Access Key ID** and **AWS Secret Access Key**
for the region and output you can skip it and put what your desired.

Reference: [AWS CLI Command Reference](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/index.html)

## AWS Secret Credentials

## Create AWS Secret Credentials

1. Login with your root user.
2. On the top right bar click on <your name>.
3. Select `Security Credentials`

![Security credentials](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_credential/security_credential.png)

4. Open the `Access Key` pane. Click on `Create New Access Key` Bottom.

![Access Key](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_credential/create_credential_edit.png)

5. You will see the **pop-up** pane that show you **Access Key ID** and **Secrect Access Key**, So Download Key File **NOW!!** because if you not donwload it now you will not be able to retrieve your secret access key again

![Secret Credentials](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/images/aws_credential/get_credential_edit.png)

**Note**: 
- You can create at most two secret credentials.
- If you want to create new one in case you already have two key. you must to delete one of them first and then you can create the new one.
- You can active/deactive your credentials. Please make sure you not want to use it before deactive or delete it.
- You must to deactive the credentials before delete it.



Reference: [Getting Your Credentials](https://docs.aws.amazon.com/sdk-for-javascript/v2/developer-guide/getting-your-credentials.html)