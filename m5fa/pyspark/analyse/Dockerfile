FROM ubuntu:19.10

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update
RUN apt-get install -y software-properties-common 
RUN add-apt-repository universe
RUN add-apt-repository multiverse
RUN apt-get update
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install -y python3-pip
RUN apt-get install -y python3.7
RUN apt-get install -y pipenv
RUN apt-get install -y wget
RUN apt-get install -y vim
RUN apt-get install -y gfortran

RUN wget -O spark.tgz https://mirror.klaus-uwe.me/apache/spark/spark-2.4.6/spark-2.4.6-bin-hadoop2.7.tgz
RUN mkdir /opt/spark
RUN tar -xzf spark.tgz --strip-components=1 -C /opt/spark
RUN rm spark.tgz

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir app
WORKDIR app

COPY Pipfile .
COPY Pipfile.lock .

ENV LANG=EN
ENV PIPENV_INSTALL_TIMEOUT=9000
RUN pipenv --python 3.7
RUN pipenv install --dev --system --deploy  

COPY *.py ./
COPY ./conf/* /opt/spark/conf/

RUN echo "export PATH=/opt/spark/bin:$PATH" >> ~/.bashrc
RUN echo "alias cdp='cd /opt/project'" >> ~/.bashrc
RUN echo "alias cdd='cd /opt/data'" >> ~/.bashrc
RUN echo "alias sps='/opt/spark/bin/spark-submit'" >> ~/.bashrc

EXPOSE 4040
