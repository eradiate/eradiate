ARG BUILDER_IMAGE
ARG BUILDER_IMAGE_VERSION
ARG BASE_IMAGE
ARG BASE_IMAGE_VERSION

FROM ${BUILDER_IMAGE}:${BUILDER_IMAGE_VERSION} as build

FROM ${BASE_IMAGE}:${BASE_IMAGE_VERSION}

RUN apt-get -q update && DEBIAN_FRONTEND="noninteractive" apt-get install -y -q wget unzip make

COPY --from=build /sources/eradiate /sources/eradiate

WORKDIR /sources/eradiate

SHELL ["conda", "run", "-n", "docker", "/bin/bash", "-c"]

ENV LD_LIBRARY_PATH=/mitsuba/ext/mitsuba2
ENV ERADIATE_SOURCE_DIR=/sources/eradiate

RUN make conda-init

RUN ertdownload

WORKDIR /app
