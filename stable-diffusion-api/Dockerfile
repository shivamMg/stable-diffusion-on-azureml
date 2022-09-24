FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# docker build -t stable-diffusion .
# docker run -it --gpus all --rm -p '5000:5000' -v $(pwd)/.cache/app:/root/.cache -v $(pwd)/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt:/models/model.ckpt -e MODEL_DIR=/models stable-diffusion

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    CONDA_DIR=/opt/conda \
    MODEL_DIR=""

SHELL ["/bin/bash", "-c"] 

RUN apt-get update && \
    apt-get install -y libglib2.0-0 wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -O ~/miniconda.sh -q --show-progress --progress=bar:force https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /code
RUN git clone https://github.com/CompVis/stable-diffusion.git && \
    conda env create -f stable-diffusion/environment.yaml && \
    conda init bash

COPY requirements.txt api/
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate ldm && \
    pip install -r api/requirements.txt

COPY api api
EXPOSE 5000 5000

COPY entrypoint.sh .
ENTRYPOINT ./entrypoint.sh