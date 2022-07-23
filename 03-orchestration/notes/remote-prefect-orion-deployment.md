# Remote Prefect Orion Deployment on AWS
---

First of all, We have to **Modifying network protocols** it default of an `AWS EC2` instance only allow TCP connection protocol with SSH on port 22. So we need to change the security protocol for run the instance if we want to be able to connect to the `Prefect` inside the instance.

1. Go to EC2 page (you need to have existing EC2 instance in this process. if not, [create it first](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/02-experiment-tracking/deploy_mlflow/deploy_mlflow_aws.md#aws-ec2-instance))
2. Go to the `Security Group`
3. Select the `Security Group ID` that your instance use that rule.

![security_group](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/03-orchestration/images/security_group.png)

4. Under Inbound rules pane, click on `Edit inbound rules` then will navigate you to other page.

![inbound_rule](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/03-orchestration/images/inbound_rule.png)

6. Click on the `Add rule` and add the following firewall rules to this group.

- HTTPS on port 443 and Anywhere IPv4
- HTTP on port 80 and Anywhere IPv4
- CUstom TCP and UDP on port 4200 and Anywhere IPv4

| **Connection Type** | **Port** |
|---------------------|----------|
| HTTP                | [80]     |
| HTTPS               | [443]    |
| TCP                 | 4200     |
| UDP                 | 4200     |

![add_rule](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/03-orchestration/images/add_rule.png)
![add_rules](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/03-orchestration/images/add_rules.png)

6. Then save rule

You will see that rules will appear under `Inbound rules` pane that you added.

![check_rules](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/03-orchestration/images/check_rules.png)

## Hosting Prefect Orion on AWS EC2 Instance

To host your `Prefect Orion` and deploy on the EC2. then after create the EC2 instance. We need to install Prefect first. To install Prefect run the code below:

```Python
pip install prefect==2.0b5
```

The code snippet above specifie to install `Prefect in version 2.0b5.
After installation step then let's set the configuration to set the public IP address of your EC2 instance.

By default `Prefect` already have the default IP address or sometime you already set others IP address before. So we can check the IP address with this command:

```Python
prefect config view
```

![prefect_config_view](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/03-orchestration/images/prefect_config_view.png)

Then let's set public IP adrress of your EC2 instance to `Prefect` with this command:

```Python
prefect config set PREFECT_ORION_UI_API_URL="http://<your-public-ip-addr>:port/api"
```

You can run the `config view` to verify that you have already set it.
If you want to unset the IP address then run this command:

```Python
prefect config unset PREFECT_ORION_UI_API_URL
```

Then start Prefect Orion with this command:

```Python
prefect orion start --host 0.0.0.0
```

## Connect to Prefect remote on AWS EC2 Instance

We have to do the same process to set configuration on the local machine with the public IP address to point to your **AWS EC2 Instance** that you already deployment the Prefect.

```Python
prefect config set PREFECT_API_URL="http://<your-public-ip-addr>:port/api"
```

Then check the configuration is done.

```Python
prefect config view
```

Now you should to see the new configuration that you set with public IP address of **AWS EC2 Instance**


