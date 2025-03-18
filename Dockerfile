# Start from the cuda base image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Install dependencies with Anaconda
RUN echo 'channels:' > /docker.yml && \
    echo '  - pytorch' >> /docker.yml && \
    echo '  - nvidia' >> /docker.yml && \
    echo '  - fastai' >> /docker.yml && \
    echo '  - bioconda' >> /docker.yml && \
    echo '  - conda-forge' >> /docker.yml && \
    echo 'dependencies:' >> /docker.yml && \
    echo '  - conda-forge::huggingface_hub=0.26' >> /docker.yml && \
    echo '  - conda-forge::toml' >> /docker.yml && \
    echo '  - conda-forge::accelerate=1.1' >> /docker.yml && \
    echo '  - conda-forge::timm=1.0' >> /docker.yml && \
    echo '  - conda-forge::pyarrow>=14.0.1' >> /docker.yml && \
    echo '  - conda-forge::pandas' >> /docker.yml && \
    echo '  - conda-forge::humanfriendly' >> /docker.yml && \
    echo '  - conda-forge::pigz' >> /docker.yml && \
    echo '  - conda-forge::tenacity' >> /docker.yml && \
    echo '  - bioconda::bbmap' >> /docker.yml && \
    echo '  - bioconda::fastp=0.24' >> /docker.yml && \
    echo '  - bioconda::sra-tools>=3' >> /docker.yml && \
    echo '  - pip' >> /docker.yml && \
    echo '  - pip:' >> /docker.yml && \
    echo '  - fastai==2.7.19' >> /docker.yml && \

# Update conda and install dependencies
RUN conda update -n base -c conda-forge conda && \
    conda env update -n base --file /docker.yml && \
    conda clean --all -f -y

# Install build dependencies and build DSK
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --recursive https://github.com/GATB/dsk.git \
    && cd dsk \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/conda .. \
    && make \
    && make install \
    && cd ../.. \
    && rm -rf dsk \
    && apt-get purge -y \
    build-essential \
    cmake \
    git \
    && apt-get autoremove -y

# Install varKoder
COPY . /varKoder
RUN conda run -n base pip install --no-deps --no-cache-dir /varKoder

# Set workdir
WORKDIR /home

# Set entrypoint
ENTRYPOINT ["conda", "run", "varKoder"]
