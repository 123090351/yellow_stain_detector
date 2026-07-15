FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    nano \
    openssh-client \
    unzip \
    vim \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && pip install \
    ultralytics \
    opencv-python-headless \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    pillow \
    pyyaml \
    tqdm

ENV YOLO_CONFIG_DIR=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib

ARG CODE_SERVER_VERSION=4.128.0
 
RUN curl -fsSL https://code-server.dev/install.sh -o /tmp/install-code-server.sh \
&& sh /tmp/install-code-server.sh \
        --method=standalone \
        --prefix=/opt/code-server \
        --version=${CODE_SERVER_VERSION} \
&& rm -f /tmp/install-code-server.sh
 
COPY docker/start-code-server.sh /usr/local/bin/start-code-server
RUN chmod +x /usr/local/bin/start-code-server
 
EXPOSE 8080

CMD ["/usr/local/bin/start-code-server"]
 
