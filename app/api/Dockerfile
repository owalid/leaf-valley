FROM python:3.8.12-slim


RUN ls -la /
RUN ls -la /home
RUN ls -la /home/api
RUN ls -la /home/api/utilities

WORKDIR /home/api

ENV PORT '5000'
ENV FLASK_ENV docker

RUN apt-get update -y
RUN apt-get install python3-opencv -y
RUN apt-get install curl -y
RUN apt-get install libgl1-mesa-glx -y
RUN pip install -U --upgrade pip
RUN pip install --upgrade setuptools pip
RUN pip install -r /home/api/requirements.txt


CMD ["python3", "main.py"]

EXPOSE 5000