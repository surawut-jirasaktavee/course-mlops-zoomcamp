# syntax=docker/dockerfile:1

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN pip3 install evidently

COPY app.py .
COPY lin_reg.bin .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=9696"]