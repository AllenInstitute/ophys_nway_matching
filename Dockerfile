FROM python:3.8.5

ARG MYBRANCH=main
ARG COMMIT="unspecified to docker build"

RUN apt-get update && \
    apt-get install -y \
      curl \
      git \
      libgl1-mesa-glx \
      libgtk2.0-dev && \
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/poetry python -

ENV PATH="/poetry/bin:${PATH}"

RUN git clone https://github.com/AllenInstitute/ophys_nway_matching && \
    cd ophys_nway_matching && \
    git checkout ${MYBRANCH} && \
    poetry install

WORKDIR /ophys_nway_matching

ENV NWAY_COMMIT_SHA=${COMMIT}

ENTRYPOINT ["poetry", "run"]
