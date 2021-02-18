FROM ubuntu:20.04

RUN apt update

RUN DEBIAN_FRONTEND=noninteractive apt install -y   \ 
    git                                             \
    cmake                                           \
    ninja-build                                     \
    clang-9                                         \
    libc++-9-dev                                    \
    libc++abi-9-dev                                 \
    libpng-dev                                      \
    zlib1g-dev                                      \
    python3.9-distutils                             \
    python3-pip                                     \
    libjpeg-dev

RUN pip3 install                \
    aabbtree                    \
    attrs                       \
    click                       \
    cerberus                    \
    dask                        \
    matplotlib                  \
    netcdf4                     \
    numpy                       \
    pint                        \
    ruamel.yaml                 \
    scipy                       \
    tinydb                      \
    tqdm                        \
    xarray                      \
    pytest                      \
    pytest-json-report          \
    pytest-metadata             \
    pytest-xdist                \
    sphinx                      \
    tabulate                    \
    black                       \
    bump2version                \
    conda-lock                  \
    ipython                     \
    ipywidgets                  \
    isort                       \
    jupyterlab                  \
    setuptools                  \
    twine                       \
    mock                        \
    pydata-sphinx-theme         \
    sphinx>=3.1                 \
    sphinxcontrib-bibtex        \
    sphinx-copybutton           \
    sphinx-gallery              \
    sphinx-panels

ENV CXX=/usr/lib/llvm-9/bin/clang++

RUN git clone --recursive https://github.com/eradiate/eradiate /sources/eradiate                \
    && cd /sources/eradiate                                                                     \
    && mkdir build                                                                              \
    && cd build                                                                                 \
    && cmake -GNinja -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") ..   \
    && ninja
