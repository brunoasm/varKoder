# Start from the cuda base image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime


# Install conda packages
RUN conda install -c fastchan fastai=2.7.12 -y && \
    conda clean --all -f -y && \
    conda install -c conda-forge timm=0.6.13 'pyarrow>=14.0.1' humanfriendly pigz tenacity -y && \
    conda install -c bioconda -c conda-forge bbmap dsk=2.3.3 fastp=0.23 -y && \
    conda clean --all -f -y

# Copy the varKoder repository contents
COPY . /varKoder
RUN conda run pip install -e /varKoder

# Set entrypoint
ENTRYPOINT ["conda", "run",  "varKoder"]

