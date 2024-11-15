#!/bin/bash
#02_constants.sh
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
    TMPDIR=$(dirname $(mktemp -u -t tmp.XXXXXXXXXX))
fi

# Determine which time command to use
if [[ "$(uname)" == "Darwin" ]]; then
    if command -v gtime &> /dev/null; then
        TIME_CMD="gtime"
    else
        TIME_CMD=""
    fi
else
    if command -v /usr/bin/time &> /dev/null; then
        TIME_CMD="/usr/bin/time"
    else
        TIME_CMD=""
    fi
fi

IM_CMD="image --seed 1 -k 7 -c 1 -m 500K -M 20M -o ./images $TESTDIR"
T1_CMD_BASE="train --overwrite --seed 2"
T2_CMD_BASE="train --overwrite --seed 3 --random-weights"
Q1_CMD="query --overwrite --include-probs --seed 4 -k 7 -c 1 -M 20M --keep-images --model trained_pretrained/trained_model.pkl fastq_query/ inferences_Bembidion"
Q2_CMD="query --overwrite --threshold 0.5 --seed 5 -k 7 -c 1 -M 20M -I inferences_Bembidion/query_images inferences_SRA"
SING_PULL="singularity pull --force varKoder.sif docker://brunoasm/varkoder"
LOCAL_PREFIX="varKoder"

# Define available architectures
declare -A ARCHITECTURES=(
    ["1"]="resnet18"
    ["2"]="resnet50"
    ["3"]="resnext101_32x8d"
    ["4"]="hf-hub:brunoasm/vit_large_patch32_224.NCBI_SRA"
)

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
DOCKER_PREFIX="docker run --platform linux/amd64 -v $TMPDIR:/tmp -v $PWD:/home brunoasm/varkoder:latest"

# Function to update prefixes based on GPU selection and time profiling choice
update_prefixes() {
    local gpu_index="$1"
    local use_time="$2"
    local time_cmd=""
    
    if [ "$use_time" = "Y" ] || [ "$use_time" = "y" ]; then
        if [ -z "$TIME_CMD" ]; then
            if [[ "$(uname)" == "Darwin" ]]; then
                echo "${color}Error: gtime not found. To use profiling on macOS, please install gnu-time:${reset}"
                echo "${color}    brew install gnu-time${reset}"
                echo "${color}Or run this script again without selecting the profiling option.${reset}"
                exit 1
            else
                echo "${color}Error: /usr/bin/time not found. Cannot enable profiling.${reset}"
                exit 1
            fi
        fi
        time_cmd="$TIME_CMD -v "
    fi

    if [ "$IS_MAC_ARM" = false ]; then
        if [ -n "$gpu_index" ]; then
            SING_PREFIX="$time_cmd$SING_PREFIX --env CUDA_VISIBLE_DEVICES=$gpu_index varKoder.sif varKoder"
            DOCKER_PREFIX="$time_cmd$DOCKER_PREFIX --gpus device=$gpu_index"
            LOCAL_PREFIX="$time_cmd env CUDA_VISIBLE_DEVICES=$gpu_index varKoder"
        else
            SING_PREFIX="$time_cmd$SING_PREFIX varKoder.sif varKoder"
            DOCKER_PREFIX="$time_cmd$DOCKER_PREFIX"
            LOCAL_PREFIX="$time_cmd varKoder"
        fi
    else
        SING_PREFIX="$time_cmd$SING_PREFIX varKoder.sif varKoder"
        DOCKER_PREFIX="$time_cmd$DOCKER_PREFIX"
        LOCAL_PREFIX="$time_cmd varKoder"
    fi
}

