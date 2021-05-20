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
    ipython                     \
    ipywidgets                  \
    isort                       \
    jupyterlab                  \
    setuptools                  \
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
    && git clone --recursive https://github.com/eradiate/eradiate.git /sources/eradiate \
    && cd /sources/eradiate \
    && wget https://eradiate.eu/data/solid_2017.zip \
    && wget https://eradiate.eu/data/us76_u86_4-spectra.zip \
    && cd resources/data && (unzip ../../solid_2017.zip || true) \
    && (unzip ../../spectra-us76_u86_4.zip || true)
RUN cd /sources/eradiate && python3 setup.py install --single-version-externally-managed --root=/ \
    && cp -r /sources/eradiate/resources /usr/local/lib/python3.8/dist-packages/resources \
    && cd / && rm -rf /sources

WORKDIR /app

ENV ERADIATE_DIR=/usr/local/lib/python3.8/dist-packages

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip
