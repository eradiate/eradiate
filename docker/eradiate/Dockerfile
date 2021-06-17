ARG BUILDER_IMAGE
ARG BUILDER_IMAGE_VERSION
ARG BASE_IMAGE
ARG BASE_IMAGE_VERSION

FROM ${BUILDER_IMAGE}:${BUILDER_IMAGE_VERSION} as build

FROM ${BASE_IMAGE}:${BASE_IMAGE_VERSION}

RUN apt-get -q update && DEBIAN_FRONTEND="noninteractive" apt-get install -y -q wget unzip make

COPY --from=build /sources/eradiate /sources/eradiate

WORKDIR /sources/eradiate

RUN wget https://eradiate.eu/data/solid_2017.zip \
    && wget https://eradiate.eu/data/us76_u86_4-spectra.zip \
    && cd resources/data && (unzip ../../solid_2017.zip || true) \
    && (unzip ../../spectra-us76_u86_4.zip || true)

ENV ERADIATE_DIR=/sources/eradiate

RUN make conda-init

WORKDIR /app
