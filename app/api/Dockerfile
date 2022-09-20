FROM python:3.8.12-slim

ENV PORT '5000'

RUN mkdir -p /home/api
COPY app/api /home/api
COPY utilities /home/api/utilities
COPY process /home/api/process
COPY utilities /utilities
COPY process /process

RUN ls -la /home/api
RUN ls -la /
RUN ls -la /utilities

WORKDIR /home/api

ENV FLASK_ENV dev
RUN apt-get update -y
RUN apt-get install python3-opencv -y
RUN apt-get install curl -y
RUN apt-get install libgl1-mesa-glx -y
RUN pip install -U --upgrade pip
RUN pip install --upgrade setuptools pip
RUN pip install -r /home/api/requirements.txt


CMD ["python3", "main.py"]

EXPOSE 5000