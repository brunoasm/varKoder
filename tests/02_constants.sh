#!/bin/bash
#varKoder test constants
TESTDIR=Bembidion
prepend_text() {
    local prefix="$1"
    shift
    eval "$@" 2> >(while read -r line; do echo "$prefix: $line"; done)
}
color=$(tput setaf 1)
reset=$(tput sgr0)
# Check if TMPDIR is not set
if [ -z "$TMPDIR" ]; then
    # TMPDIR is not set, so create a temporary directory and assign it
    TMPDIR=$(dirname $(mktemp -u -t tmp.XXXXXXXXXX))
fi
IM_CMD="image --seed 1 -k 7 -c 1 -m 500K -M 20M -o ./images $TESTDIR"
T1_CMD="train --overwrite --seed 2 -e 0 -z 5 ./images ./trained_pretrained"
T2_CMD="train --overwrite --architecture resnet18 --seed 3 --random-weights -e 5 -z 5 ./images ./trained_random"
Q1_CMD="query --overwrite --include-probs --seed 4 -k 7 -c 1 -M 20M --keep-images --model trained_pretrained/trained_model.pkl fastq_query/ inferences_Bembidion"
Q2_CMD="query --overwrite --threshold 0.5 --seed 5 -k 7 -c 1 -M 20M -I inferences_Bembidion/query_images inferences_SRA"
SING_PULL="singularity pull --force varKoder.sif docker://brunoasm/varkoder"
LOCAL_PREFIX="varKoder"

# Detect if the system is a Mac ARM
IS_MAC_ARM=false
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    IS_MAC_ARM=true
fi

# Function to get available GPU indices (only for non-Mac ARM systems)
get_gpu_indices() {
    if [ "$IS_MAC_ARM" = false ] && command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '
    else
        echo ""
    fi
}

# Get available GPU indices
AVAILABLE_GPUS=$(get_gpu_indices)

# Function to set CUDA_VISIBLE_DEVICES (only for non-Mac ARM systems)
set_cuda_visible_devices() {
    if [ "$IS_MAC_ARM" = false ]; then
        if [ -n "$1" ]; then
            export CUDA_VISIBLE_DEVICES="$1"
            echo "Using GPU index: $1"
        else
            unset CUDA_VISIBLE_DEVICES
            echo "No GPU selected. Using CPU."
        fi
    fi
}

SING_PREFIX="singularity exec --no-home --cleanenv --nv -B ${TMPDIR}:/tmp -B ${PWD}:/home --pwd /home"
DOCKER_PREFIX="docker run --platform linux/amd64 -v $TMPDIR:/tmp -v $PWD:/home"

# Function to update prefixes based on GPU selection (only for non-Mac ARM systems)
update_prefixes() {
    if [ "$IS_MAC_ARM" = false ]; then
        local gpu_index="$1"
        if [ -n "$gpu_index" ]; then
            SING_PREFIX="$SING_PREFIX --env CUDA_VISIBLE_DEVICES=$gpu_index varKoder.sif varKoder"
            DOCKER_PREFIX="$DOCKER_PREFIX --gpus device=$gpu_index brunoasm/varkoder:latest varKoder"
            LOCAL_PREFIX="CUDA_VISIBLE_DEVICES=$gpu_index varKoder"
        else
            SING_PREFIX="$SING_PREFIX varKoder.sif varKoder"
            DOCKER_PREFIX="$DOCKER_PREFIX brunoasm/varkoder:latest varKoder"
            LOCAL_PREFIX="varKoder"
        fi
    else
        SING_PREFIX="$SING_PREFIX varKoder.sif varKoder"
        DOCKER_PREFIX="$DOCKER_PREFIX brunoasm/varkoder:latest varKoder"
        LOCAL_PREFIX="varKoder"
    fi
}
