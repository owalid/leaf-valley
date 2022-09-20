FROM python:3.8.12-slim

ENV PORT '5000'

RUN mkdir -p /home/api
COPY app/api /home/api
COPY utilities /utilities
COPY process /process

WORKDIR /home/api

ENV FLASK_ENV = 'dev'
RUN apt-get update -y
RUN apt-get install python3-opencv -y
RUN apt-get install curl -y
RUN apt-get install libgl1-mesa-glx -y
RUN pip install -U --upgrade pip
RUN pip install --upgrade setuptools pip
RUN pip install -r /home/api/requirements.txt

EXPOSE 5000