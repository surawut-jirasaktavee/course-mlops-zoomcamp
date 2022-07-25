# Create Environment
---
## Create virtual environment for deployment

I have used `Pipenv`. `Pipenv` is recommended as it’s a higher-level tool that simplifies dependency management for common use cases.

Use pip to install Pipenv:

```Python
pip install --user pipenv
```

Then to inspect the version of the packages that you use in your project let's check with `pip freeze`

```Python
pip freeze | grep <module>
```

Now you know the specific version of your packages on each module let's create an environment. 

For example:

```Python
pipenv install scikit-learn==1.1.1 flask --python==3.9.12
```

This virtual environment will be isolated from the base python environment that we have.
To run activate this project's virtual environment, run this command:

```Python
pipenv shell
```

And run a command inside the virtual environment with:

```Python
pipenv run
```

**Additional**: We can install dependency only in the dev environment for test and not affect the deploy environment we can create virtual environment with this command:

```Python
pipenv install --dev <module-name> 
```

Reference: [pipenv](https://docs.python-guide.org/dev/virtualenvs/#virtualenvironments-ref)
