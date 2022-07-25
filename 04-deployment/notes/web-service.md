# Deploy the Machine Learning model using Flask and Docker
---

To create `web service` to deploy the machine learning model using `flask` framework in python.

```Python
from flask import Flask, requests, jsonify

app = Flask(__name__) # put your app name instead of __name__

@app.route("/endpoint", methods=["POST"]) # put your endpoint of your name and the method
def func():
    result = "prediction"
    return jsonify(result) # to return json object

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
```

In the above code snippet:

1. Import the **Flask** class. An instance of this class will be our application.
2. Create an instance of this class. This first argument is the name of the application's module or package. `__name___` is a convenient shortcut for this that is appropriate for most cases. This is needed so that Flask knows where to look for resources such as templates and static files.
3. Then use the **route()** decorator to tell Flask what URL should trigger our function.
4. The function will return the message we want to display to user's broswer or anything else.
5. Run the application with `app.run()` with open debug mode, host, and port number.

This case is use only in the development environment. Not for the production environment.
You will see the message below the warn you about the development reason

`WARNING: This is a development server. Do not use it in a production deployment.`

In case you want to use the model in the production environment then we need to install the web service.
To install run the command.

```Python
pip install gunicorn
```

And run the command to deploy the model to `Flask web service with gunicorn`

```Python
gunicorn --bind=0.0.0.0:9696 func:app
```

The only required argument to `Gunicorn` tells it how to load your Flask application. The syntax is {module_import}:{app_variable}. module_import is the dotted import name to the module with your application. app_variable is the variable with the application. The code above equivalent to from the endpoint import app.

To create docker image. first let's create `Dockerfile`. inside the Dockerfile:

```Dockerfile
FROM python:3.9-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY [ "predict.py", "LinearRegression_2022-06-18 17:37:50.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn",  "--bind=0.0.0.0:9696",  "predict:app"]
```

From the `Dockerfile`:

1. Specifying the Python version that I used in my environment.
2. Run the pip command to update `pip` to make sure we have the latest `pip` version.
3. Run the command to install `pipenv` virtual environment.
4. I have specify working directory of my app.
5. Copy the necessary file e.g. `Pipfile`, `Pipfile.lock` from current directory.
6. Run the command to installation.
7. Copy the python script and the model from current directory.
8. Expose the port number `9696`.
9. Specify the entrypoint to the `Gunicorn` at port to map localhost with port 9696 to use the application.

So you can create docker image by run the command:

```zsh
docker build -t <image-name>:<tag>
```

Then run the docker.

```zsh
docker run -it --rm -p 9696:9696 <image-name>:<tag>
```

Now we can use the model in the purpose of model.