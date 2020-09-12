FROM python:3.8.5

ARG MYBRANCH=master

RUN apt-get update && \
    apt-get install -y \
      git \
      libgl1-mesa-glx \
      libgtk2.0-dev

RUN git clone https://github.com/AllenInstitute/ophys_nway_matching

RUN cd ophys_nway_matching && \
    git checkout ${MYBRANCH} && \
    pip install .

WORKDIR /ophys_nway_matching