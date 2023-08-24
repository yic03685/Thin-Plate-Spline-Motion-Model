FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Make python 3.10 as default and create an environment
RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-apt \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wget \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libsm6 \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libxext6 \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libxrender-dev \     
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

# install conda because of pytorch3d
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /workspace

RUN conda create --name training -y python=3.9

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "training", "/bin/bash", "-c"]

RUN git clone https://github.com/yic03685/Thin-Plate-Spline-Motion-Model.git
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia


WORKDIR /workspace/Thin-Plate-Spline-Motion-Model
RUN pip install -r requirements.txt
RUN pip install -U accelerate
RUN pip install bitsandbytes

ENV PYTORCH_CUDA_ALLOC_CONF "max_split_size_mb:512"