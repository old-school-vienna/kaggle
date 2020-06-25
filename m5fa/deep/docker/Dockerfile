FROM tensorflow/tensorflow:2.2.0

RUN apt-get update
RUN apt-get install -y openjdk-8-jdk-headless
RUN apt-get install -y wget
RUN apt-get install -y git

RUN wget -O spark.tgz https://mirror.klaus-uwe.me/apache/spark/spark-3.0.0-preview2/spark-3.0.0-preview2-bin-hadoop2.7.tgz
RUN mkdir /opt/spark
RUN tar -xzf spark.tgz --strip-components=1 -C /opt/spark
RUN rm spark.tgz

RUN python -m pip install --upgrade pip
RUN pip install Pandas==1.0.5
RUN pip install pyspark==3.0.0
RUN pip install pytest
RUN pip install git+https://github.com/tensorflow/docs

RUN mkdir /opt/project
WORKDIR /opt/project
