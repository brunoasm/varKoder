# Start from the cuda base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install dependencies
COPY ./conda_environments/docker.yml docker.yml

RUN conda env update -n base --file docker.yml && \
    conda clean --all -f -y

# Install varKoder
COPY . /varKoder

RUN conda run pip install --no-deps --no-cache-dir -e /varKoder

#Set workdir
WORKDIR /home

# Set entrypoint
ENTRYPOINT ["conda", "run",  "varKoder"]

