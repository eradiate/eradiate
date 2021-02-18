ARG REGISTRY=fxia
ARG VERSION=v1.0.0

FROM ${REGISTRY}/eradiate:${VERSION}

RUN pip3 install jupyterlab

ENV PORT=8888

CMD jupyter lab --no-browser --allow-root --port=$PORT --ip='0.0.0.0' --allow_origin='*'

EXPOSE 8888
