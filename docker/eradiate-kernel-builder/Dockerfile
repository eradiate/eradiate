ARG BASE_IMAGE
ARG BASE_IMAGE_VERSION

FROM ${BASE_IMAGE}:${BASE_IMAGE_VERSION}

ARG ERADIATE_KERNEL_VERSION

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y   \
    cmake                                           \
    git                                             \
    ninja-build                                     \
    clang-9                                         \
    libc++-9-dev                                    \
    libc++abi-9-dev                                 \
    libpng-dev                                      \
    zlib1g-dev                                      \
    libjpeg-dev

ENV CC=clang-9
ENV CXX=clang++-9

COPY . /sources/eradiate
WORKDIR /sources/eradiate
RUN make conda-init

WORKDIR /build/eradiate-kernel
RUN cmake -GNinja -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") /sources/eradiate/ext/mitsuba2
RUN ninja

