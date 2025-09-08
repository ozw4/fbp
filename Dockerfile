# syntax=docker/dockerfile:1


FROM nvcr.io/nvidia/pytorch:24.09-py3 AS develop

ARG USERNAME=dcuser
ARG UID=1000
ARG GID=1000

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
#ENV CFLAGS="-w" CXXFLAGS="-w"
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    DEBIAN_FRONTEND=noninteractive

RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

RUN --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    fontconfig \
    ttf-mscorefonts-installer \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN fc-cache -fv

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install torchaudio==2.7.0

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=bind,source=.devcontainer/requirements-dev.txt,target=requirements-dev.txt \
    python -m pip install -r requirements-dev.txt

RUN addgroup --gid $GID $USERNAME && \
    adduser --disabled-password --gecos "" --shell "/sbin/nologin" --uid $UID --gid $GID $USERNAME

USER $USERNAME

COPY ruff.toml /home/$USERNAME
ENV PYTHONPATH="/workspace:${PYTHONPATH}"
WORKDIR /workspace

CMD ["/bin/bash"]
