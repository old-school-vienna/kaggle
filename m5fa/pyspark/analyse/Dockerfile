FROM ubuntu:19.10

#RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get update
RUN apt-get install -y openjdk-8-jdk
RUN apt install -y python3-pip
RUN apt-get install -y python3.7
RUN apt-get install -y pipenv

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir app
WORKDIR app

COPY Pipfile .
COPY Pipfile.lock .

ENV LANG=EN
RUN pipenv --python 3.7
RUN pipenv install --dev --system --deploy  

COPY *.py ./
