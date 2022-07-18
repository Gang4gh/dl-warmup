FROM tensorflow/tensorflow:2.4.1-gpu

LABEL Description="A docker image for the develop environment of dsum project based on tensorflow 2.X"

RUN apt update && apt install -y sudo git screen vim p7zip-full libxml-parser-perl
RUN pip install tensorflow_datasets git+https://github.com/tagucci/pythonrouge.git

ENV PYTHONPATH=/dsum/tensorflow/models
RUN git clone --depth 1 https://github.com/tensorflow/models.git /dsum/tensorflow/models && cd /dsum/tensorflow/models && rm -rf .git* research samples tutorials

RUN sed -i 's,^path-exclude=/usr/share/man/,#path-exclude=/usr/share/man/,' /etc/dpkg/dpkg.cfg.d/excludes
RUN apt update && apt install -y --reinstall man manpages manpages-dev coreutils sudo screen gzip git-man
RUN pip install --upgrade pip && pip install flake8==2.5.4

#RUN useradd -m -G sudo dsumuser && echo dsumuser:gg | chpasswd
#USER dsumuser
