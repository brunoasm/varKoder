# Start from the cuda base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Copy the varKoder repository contents
COPY . /varKoder

# Install conda packages
RUN conda install -n base mamba && \
    mamba env update -n base --file /varKoder/conda_environments/docker.yml && \
    mamba clean --all -f -y && \ 
    conda run pip install -e /varKoder

# Set entrypoint
ENTRYPOINT ["conda", "run",  "varKoder"]

