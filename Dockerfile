# Start from the cuda base image
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Install wget for downloading Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /miniconda && \
    rm /miniconda.sh

# Set path to include conda
ENV PATH="/miniconda/bin:${PATH}"

# Create the conda environment and install dependencies available via conda
RUN conda create -n varKoder -c pytorch -c nvidia pytorch=2.0 pytorch-cuda=11.7 -y && \
    conda install -n varKoder -c fastai fastai=2.7.12 -y && \
    conda install -n varKoder -c conda-forge timm=0.6.13 pyarrow pandas humanfriendly pigz tenacity -y && \
    conda install -n varKoder -c bioconda bbmap dsk=2.3.3 fastp=0.23 -y && \
    conda clean --all -f -y

# Set up the conda environment as default and initialize Conda
ENV PATH="/miniconda/bin:${PATH}"
RUN echo "source /miniconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate varKoder" >> ~/.bashrc

# Copy the varKoder repository contents
COPY . /varKoder

# Install varKoder in editable mode within the Conda environment
RUN /miniconda/bin/conda run -n varKoder pip install -e /varKoder
