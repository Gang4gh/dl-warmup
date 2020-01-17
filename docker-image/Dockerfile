FROM tensorflow/tensorflow:2.1.0-gpu-py3

LABEL Description="A docker image for the develop environment of dsum project atop of tensorflow 2.X"

RUN apt update && apt install -y sudo git screen vim p7zip-full libxml-parser-perl
RUN pip install tensorflow_datasets git+https://github.com/tagucci/pythonrouge.git

ENV PYTHONPATH=/dsum/tensorflow/models
RUN git clone --depth 1 https://github.com/tensorflow/models.git /dsum/tensorflow/models && cd /dsum/tensorflow/models && rm -rf .git* research samples tutorials

RUN useradd -m -G sudo dsumuser && echo dsumuser:gg | chpasswd
USER dsumuser
