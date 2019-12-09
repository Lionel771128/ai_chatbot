#FROM python:3.6.8
#LABEL maintainer = lionel771128
#ENV PYTHONUNBUFFERED 1
#RUN mkdir /djangodemo_docker_test
#WORKDIR /djangodemo_docker_test
#COPY . /djangodemo_docker_test
#RUN pip install -r requirements.txt

FROM python:3.6.8
LABEL maintainer = lionel771128
ENV PYTHONUNBUFFERED 1
RUN mkdir /docker_api
WORKDIR /docker_api
COPY . /docker_api
RUN pip install -r requirements.txt
RUN python manage.py migrate


