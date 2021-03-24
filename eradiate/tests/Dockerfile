ARG REGISTRY=fxia
ARG VERSION=v1.0.0

FROM ${REGISTRY}/eradiate-kernel:${VERSION}

RUN apt-get update && apt-get install -y python3 python3-pip git wget unzip

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
    iapws                       \
    ipython                     \
    ipywidgets                  \
    isort                       \
    jupyterlab                  \
    setuptools                  \
    toolz                       \
    twine                       \
    mock                        \
    pydata-sphinx-theme         \
    sphinx                      \
    sphinxcontrib-bibtex        \
    sphinx-copybutton           \
    sphinx-gallery              \
    sphinx-panels               \
    pinttrs                     \
    iapws

ENV ERADIATE_DIR=/sources/eradiate

RUN mkdir -p /sources \
    && git clone --recursive https://github.com/eradiate/eradiate.git /sources/eradiate

WORKDIR /sources/eradiate

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

CMD echo "Downloading datasets..." && wget https://eradiate.eu/data/solid_2017.zip -q  \
    && wget https://eradiate.eu/data/spectra-us76_u86_4.zip -q \
    && echo "Done downloading." \
    && cd resources/data && (unzip ../../solid_2017.zip || true) \
    && (unzip ../../spectra-us76_u86_4.zip || true) && cd ../.. \
    && pytest eradiate