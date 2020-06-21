FROM tensorflow/tensorflow:2.2.0

RUN apt-get update
RUN apt-get install -y libsnappy-dev
RUN apt-get install -y openjdk-8-jdk-headless

RUN python -m pip install --upgrade pip
RUN pip install Pandas
RUN pip install packaging
RUN pip install fastparquet
RUN pip install python-snappy
RUN pip install pyspark
RUN pip install pytest
