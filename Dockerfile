FROM python:3.11.4-slim

RUN apt update && apt upgrade -y
RUN apt-get update && apt-get install -y sudo
RUN pip install --upgrade pip