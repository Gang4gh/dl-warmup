FROM tensorflow/tensorflow:2.0.0-gpu-py3

LABEL Description="This image is used to develop the dsum model atop tensorflow 2.0"

RUN apt update && apt install -y git screen vim
RUN pip install tensorflow_datasets git+https://github.com/tagucci/pythonrouge.git

ENV PYTHONPATH=/dsum/tensorflow/models
RUN git clone --depth 1 https://github.com/tensorflow/models.git /dsum/tensorflow/models && cd /dsum/tensorflow/models && rm -rf .git* research samples tutorials

# fix a code issue
RUN cd $PYTHONPATH/official/modeling/activations/ && grep -vwE 'tf.keras.utils.register_keras_serializable' gelu.py > tmp && mv tmp gelu.py
RUN cd $PYTHONPATH/official/modeling/activations/ && grep -vwE 'tf.keras.utils.register_keras_serializable' swish.py > tmp && mv tmp swish.py

RUN apt update && apt install -y sudo
RUN useradd -m -G sudo dsumuser && echo dsumuser:gg | chpasswd
USER dsumuser
