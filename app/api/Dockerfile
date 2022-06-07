FROM python:3.8.12-slim-buster

ENV PORT '5000'

RUN mkdir -p /home/api
COPY . /home/api
WORKDIR /home/api

RUN apt-get update
RUN apt-get install -y gcc python3-dev musl-dev libpq-dev
RUN pip install --upgrade pip
# RUN pip install --upgrade setuptools wheel cython
RUN pip install -r config/requirements.txt

EXPOSE 5000