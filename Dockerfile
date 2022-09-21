FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# docker build -t stable-diffusion .
# docker run -it --gpus all --rm -p '5000:5000' -v $(pwd)/.cache/app:/root/.cache -v $(pwd)/models/ldm/stable-diffusion-v1/model.ckpt:/models/model.ckpt -e MODEL_DIR=/models stable-diffusion

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    CONDA_DIR=/opt/conda \
    MODEL_DIR=""

RUN apt-get update && \
    apt-get install -y libglib2.0-0 wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget -O ~/miniconda.sh -q --show-progress --progress=bar:force https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /code
RUN git clone https://github.com/CompVis/stable-diffusion.git && \
    cd stable-diffusion && \
    conda env create -f environment.yaml
SHELL ["/bin/bash", "-c"] 
# RUN echo "source activate ldm" > /root/.bashrc

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate ldm && \
    pip install gunicorn==20.1.0 flask==2.2.2

COPY ./scripts/gunicorn.conf.py ./stable-diffusion/scripts/
COPY ./scripts/myserver.py ./stable-diffusion/scripts/
COPY ./scripts/mytxt2img.py ./stable-diffusion/scripts/
COPY ./scripts/wsgi.py ./stable-diffusion/scripts/
COPY ./entrypoint.sh ./stable-diffusion/

EXPOSE 5000 5000

WORKDIR /code/stable-diffusion
ENTRYPOINT ./entrypoint.sh