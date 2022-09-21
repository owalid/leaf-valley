FROM python:3.9.12-slim-buster


RUN mkdir -p /home/api
COPY app/api /home/api/
COPY utilities /home/api/utilities
COPY process /home/api/process

RUN ls -la /
RUN ls -la /home
RUN ls -la /home/api
RUN ls -la /home/api/utilities

WORKDIR /home/api

ENV PORT '5000'
ENV FLASK_ENV docker

RUN apt-get update -y
RUN apt-get install gcc python3-dev musl-dev libpq-dev -y
RUN apt-get install g++ jpeg-dev zlib-dev libjpeg make
RUN apt-get install python3-opencv -y
RUN apt-get install curl -y
RUN apt-get install libgl1-mesa-glx -y
RUN pip install -U --upgrade pip
RUN pip install --upgrade setuptools pip
RUN pip install -r /home/api/requirements.txt
RUN pip install numpy --upgrade
RUN pip install matplotlib --upgrade


CMD ["python", "main.py"]

EXPOSE 5000