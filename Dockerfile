# Use BuildKit features
# syntax=docker/dockerfile:1.4

ARG PEXECS=1
ARG EXPERIMENTS
ARG SUITES
ARG MEASUREMENTS

FROM debian:latest as base

FROM base as build
WORKDIR /app

ARG FULL
ARG PEXECS
ARG EXPERIMENTS
ARG SUITES
ARG MEASUREMENTS

ENV FULL=$FULL
ENV PEXECS=$PEXECS
ENV EXPERIMENTS=$EXPERIMENTS
ENV SUITES=$SUITES
ENV MEASUREMENTS=$MEASUREMENTS

COPY packages-debian.txt .

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
    autotools-dev bison bc perl build-essential clang cmake curl elfutils \
    flex gdb git gperf fontconfig libfontconfig1-dev libboost-all-dev \
    libdwarf-dev libdw-dev libedit-dev libffi-dev liblzma-dev libssl-dev \
    libtinfo-dev libunwind-dev libzip-dev libzstd-dev \
    libx11-6 libx11-xcb1 libxkbcommon-x11-dev libxext6 libxrandr2 libxcb1 \
    libxrender1 libxcursor1 libgl1 libgl1-mesa-dri libglx-mesa0 x11-apps \
    llvm-dev make ninja-build pkg-config pipx procps python3 rsync time \
    xvfb xauth which zlib1g-dev zstd \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

ENV CARGO_HOME=/cargo
ENV RUSTUP_HOME=/rustup
ENV PATH=$CARGO_HOME/bin:$PATH

RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly

RUN rustc --version && cargo --version

COPY pyproject.toml .
COPY Makefile .
RUN make venv


COPY src src
COPY *.py .

RUN if [ "$FULL" = "false" ]; then \
    make download-bins; \
    else \
    RUN make build-heaptrack; \
    RUN make build-alloy; \
    RUN make build-benchmarks; \
    fi
#
COPY extra extra

FROM build as runtime

ARG BUILD_QUICK=false
ARG PEXECS
ARG EXPERIMENTS
ARG SUITES
ARG MEASUREMENTS

ENV BUILD_QUICK=$BUILD_QUICK
ENV PEXECS=$PEXECS
ENV EXPERIMENTS=$EXPERIMENTS
ENV SUITES=$SUITES
ENV MEASUREMENTS=$MEASUREMENTS

WORKDIR /app


CMD make run-benchmarks


