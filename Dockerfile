# Start from the cuda base image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Create a single RUN command for minimal conda operations
RUN conda update -n base -c conda-forge conda && \
    # Configure conda channels for bioinformatics tools
    conda config --add channels bioconda && \
    conda config --add channels conda-forge && \
    # Install only bioinformatics tools and specialized packages via conda
    conda install -y \
        bioconda::bbmap \
        bioconda::fastp=0.24 \
        bioconda::sra-tools>=3 \
        conda-forge::pigz \
        pip && \
    # Clean conda caches
    conda clean --all -f -y && \
    # Create a pip.conf to disable cache
    mkdir -p /root/.config/pip && \
    echo "[global]" > /root/.config/pip/pip.conf && \
    echo "no-cache-dir = true" >> /root/.config/pip/pip.conf

# Install pip packages separately (most packages are available via pip)
RUN pip install --no-cache-dir \
    fastai==2.7.19 \
    "huggingface_hub~=0.29.0" \
    toml \
    "accelerate~=1.5.0" \
    "timm~=1.0.0" \
    pyarrow>=14.0.1 \
    pandas \
    humanfriendly \
    tenacity

# Install build dependencies, build DSK, and clean up in a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        && git clone --recursive https://github.com/GATB/dsk.git \
        && cd dsk \
        && mkdir build \
        && cd build \
        && cmake -DCMAKE_INSTALL_PREFIX=/opt/conda .. \
        && make \
        && make install \
        && cd ../.. \
        && rm -rf dsk \
        # Clean up build dependencies
        && apt-get purge -y \
            build-essential \
            cmake \
            git \
        && apt-get autoremove -y \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Install varKoder
COPY . /varKoder
RUN pip install --no-deps /varKoder

# Set workdir
WORKDIR /home

# Set entrypoint
ENTRYPOINT ["conda", "run", "varKoder"]
