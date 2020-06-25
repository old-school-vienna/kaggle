FROM dockerregistry.seu.test.sozvers.at/ibmcom/tensorflow-ppc64le

RUN apt-get update
RUN apt-get install -y libsnappy-dev
RUN apt-get install -y openjdk-8-jdk-headless
RUN apt-get install -y wget

RUN wget -O spark.tgz https://mirror.klaus-uwe.me/apache/spark/spark-3.0.0-preview2/spark-3.0.0-preview2-bin-hadoop2.7.tgz
RUN mkdir /opt/spark
RUN tar -xzf spark.tgz --strip-components=1 -C /opt/spark
RUN rm spark.tgz


RUN python -m pip install --upgrade pip
RUN pip install Pandas==1.0.5
RUN pip install packaging==20.4
# RUN pip install fastparquet==0.4.0
# RUN pip install python-snappy==0.5.4
RUN pip install pyspark==3.0.0
RUN pip install pytest

RUN mkdir /opt/project
WORKDIR /opt/project
