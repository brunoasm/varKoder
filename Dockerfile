# Start from the cuda base image
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Install dependencies with Anaconda
RUN echo 'channels:' > /docker.yml && \
    echo '  - pytorch' >> /docker.yml && \
    echo '  - nvidia' >> /docker.yml && \
    echo '  - fastai' >> /docker.yml && \
    echo '  - bioconda' >> /docker.yml && \
    echo '  - conda-forge' >> /docker.yml && \
    echo 'dependencies:' >> /docker.yml && \
    echo '  - fastai::fastai=2.7.13' >> /docker.yml && \
    echo '  - conda-forge::huggingface_hub=0.23' >> /docker.yml && \
    echo '  - conda-forge::accelerate=0.31' >> /docker.yml && \
    echo '  - conda-forge::timm=1.0' >> /docker.yml && \
    echo '  - conda-forge::pyarrow>=14.0.1' >> /docker.yml && \
    echo '  - conda-forge::pandas' >> /docker.yml && \
    echo '  - conda-forge::humanfriendly' >> /docker.yml && \
    echo '  - conda-forge::pigz' >> /docker.yml && \
    echo '  - conda-forge::tenacity' >> /docker.yml && \
    echo '  - bioconda::bbmap' >> /docker.yml && \
    echo '  - bioconda::dsk=2.3.3' >> /docker.yml && \
    echo '  - bioconda::fastp=0.23' >> /docker.yml && \
    echo '  - bioconda::sra-tools>=3' >> /docker.yml

RUN conda update -n base -c defaults conda && \
    conda env update -n base --file /docker.yml && \
    conda clean --all -f -y

# Install varKoder
COPY . /varKoder

RUN conda run -n base pip install --no-deps --no-cache-dir /varKoder

#Set workdir
WORKDIR /home

# Set entrypoint
ENTRYPOINT ["conda", "run",  "varKoder"]

