ARG BUILDER_IMAGE
ARG BUILDER_IMAGE_VERSION
ARG BASE_IMAGE
ARG BASE_IMAGE_VERSION

FROM ${BUILDER_IMAGE}:${BUILDER_IMAGE_VERSION} as build

FROM ${BASE_IMAGE}:${BASE_IMAGE_VERSION}

ARG ERADIATE_KERNEL_VERSION

COPY --from=build /build/eradiate-kernel/eradiate-kernel-dist_${ERADIATE_KERNEL_VERSION}.tar /build/mitsuba/

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y ninja-build cmake libc++-9-dev libz-dev libpng-dev libjpeg-dev libxrandr-dev libxinerama-dev libxcursor-dev


RUN mkdir -p /build/mistuba /mitsuba && cd /build/mitsuba && tar -xf eradiate-kernel-dist_${ERADIATE_KERNEL_VERSION}.tar  \
    && mv dist /mitsuba/dist \
    && cd / && rm -rf /build

ENV MITSUBA_DIR=/mitsuba
WORKDIR /app

ENV PYTHONPATH="$MITSUBA_DIR/dist/python:$MITSUBA_DIR/$BUILD_DIR/dist/python:$PYTHONPATH"
ENV PATH="$MITSUBA_DIR/dist:$MITSUBA_DIR/$BUILD_DIR/dist:$PATH"

CMD mitsuba --help