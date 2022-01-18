ARG BUILDER_IMAGE
ARG BUILDER_IMAGE_VERSION
ARG BASE_IMAGE
ARG BASE_IMAGE_VERSION

FROM ${BUILDER_IMAGE}:${BUILDER_IMAGE_VERSION} as build

FROM ${BASE_IMAGE}:${BASE_IMAGE_VERSION}

ARG ERADIATE_KERNEL_VERSION

COPY --from=build /build/eradiate-kernel /mitsuba

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y ninja-build cmake libc++-9-dev libz-dev libpng-dev libjpeg-dev libxrandr-dev libxinerama-dev libxcursor-dev llvm-9

ENV MITSUBA_DIR=/mitsuba/ext/mitsuba2
WORKDIR /app

ENV PYTHONPATH="$MITSUBA_DIR/python"
ENV PATH="$MITSUBA_DIR:$PATH"

CMD mitsuba --help