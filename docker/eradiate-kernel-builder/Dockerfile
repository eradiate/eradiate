ARG BASE_IMAGE
ARG BASE_IMAGE_VERSION

FROM ${BASE_IMAGE}:${BASE_IMAGE_VERSION}

ARG ERADIATE_KERNEL_VERSION
ARG NUM_CORES=4

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y   \
    git                                             \
    ninja-build                                     \
    clang-9                                         \
    libc++-9-dev                                    \
    libc++abi-9-dev                                 \
    libpng-dev                                      \
    zlib1g-dev                                      \
    build-essential                                 \
    libjpeg-dev

RUN conda install -y cmake

ENV CC=clang-9
ENV CXX=clang++-9

COPY . /sources/eradiate
WORKDIR /sources/eradiate
RUN make conda-init

WORKDIR /build/eradiate-kernel
RUN cmake --preset default -S /sources/eradiate -B .
RUN cmake --build . -j${NUM_CORES}

