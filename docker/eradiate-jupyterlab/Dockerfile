ARG BASE_IMAGE
ARG BASE_IMAGE_VERSION

FROM ${BASE_IMAGE}:${BASE_IMAGE_VERSION}

ENV PORT=8888
ENV ERADIATE_SOURCE_DIR=/sources/eradiate

CMD jupyter lab --no-browser --allow-root --port=$PORT --ip='0.0.0.0' --allow_origin='*'
EXPOSE 8888
