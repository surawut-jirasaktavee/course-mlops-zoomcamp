FROM python:3.9-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY [ "predict.py", "LinearRegression_2022-06-18 17:37:50.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn",  "--bind=0.0.0.0:9696",  "predict:app"]

