#nvidia-docker build -t VDSR4Geo ./
#NV_GPU=0 nvidia-docker run -it -v /home:/home/ --name VDSR_GPU0 VDSR4Geo

FROM nvidia/cuda:9.0-cudnn7-devel

MAINTAINER Jake Shermeyer

# Pick up some TF dependencies
RUN add-apt-repository -y ppa:ubuntugis/ppa
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python3 \
        python3-dev \
        python-pip \
        python3-pip \
        python3-setuptools \
        libproj-dev libgdal-dev libgeos-dev libjson0-dev vim python-gdal \
        python3-gdal \
        rsync \
        software-properties-common \
        unzip \
        python-h5py \
    	python-yaml \
    	python-pydot \
        python3-h5py \
    	python3-yaml \
    	python3-pydot \
	    graphviz \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        scikit-learn \
        scikit-image \
        pandas \
        opencv-python \
        tqdm \
        argparse \
        Pillow \

        && \
    python3 -m ipykernel.kernelspec


# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# These lines will be edited automatically by parameterized_docker_build.sh. #
# COPY _PIP_FILE_ /
# RUN pip --no-cache-dir install /_PIP_FILE_
# RUN rm -f /_PIP_FILE_

# Install TensorFlow GPU version from central repo
RUN pip3 --no-cache-dir install --upgrade tensorflow-gpu
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #

# RUN ln -s /usr/bin/python3 /usr/bin/python#


# TensorBoard
EXPOSE 5900
# IPython
EXPOSE 8888

WORKDIR "/"

# TensorBoard
RUN ["/bin/bash"]

