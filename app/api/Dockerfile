FROM python:3.8.12-alpine

ENV PORT '5000'

RUN mkdir -p /home/api
COPY . /home/api
WORKDIR /home/api

RUN apk update
RUN apk add gcc python3-dev musl-dev libpq-dev curl
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r /home/api/requirements.txt

EXPOSE 5000