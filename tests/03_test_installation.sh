#!/bin/bash
# 03_test_installation.sh
source 02_constants.sh

echo "$color What would you like to test? Please choose an option by typing the number:"
echo "1. local conda installation (default)"
echo "2. docker image"
echo "3. singularity image"
echo "Press Enter to use default [1] or type your choice:$reset"
read -p "Enter your choice [1]: " choice
choice=${choice:-1}

# GPU selection (skip for Mac ARM)
if [ "$IS_MAC_ARM" = false ]; then
    if [ -n "$AVAILABLE_GPUS" ]; then
        echo "$color Available GPUs: $AVAILABLE_GPUS"
        echo "Enter the index of the GPU you want to use, or press Enter to use CPU:$reset"
        read -p "GPU index: " gpu_index
        set_cuda_visible_devices "$gpu_index"
    else
        echo "$color No NVIDIA GPUs detected. Using CPU.$reset"
        gpu_index=""
    fi
else
    echo "$color Mac ARM system detected. GPU selection is not applicable.$reset"
    gpu_index=""
fi

# Ask about time profiling
echo "$color Would you like to profile resource usage with /usr/bin/time?"
echo "Press Enter for default [N] or type Y for yes:$reset"
read -p "Use time profiling? [N]: " use_time
use_time=${use_time:-N}

# Ask about neural network architecture
echo "$color Choose neural network architecture:"
echo "1. resnet18 (default)"
echo "2. resnet50"
echo "3. resnext101_32x8d"
echo "4. hf-hub:brunoasm/vit_large_patch32_224.NCBI_SRA"
echo "Press Enter to use default [1] or type your choice (1-4):$reset"
read -p "Enter choice [1]: " arch_choice
arch_choice=${arch_choice:-1}

case $arch_choice in
    1|"")
        ARCHITECTURE=$ARCH1
        ;;
    2)
        ARCHITECTURE=$ARCH2
        ;;
    3)
        ARCHITECTURE=$ARCH3
        ;;
    4)
        ARCHITECTURE=$ARCH4
        ;;
    *)
        echo "Invalid choice. Using default resnet18"
        ARCHITECTURE=$ARCH1
        ;;
esac

# Ask about epochs
echo "$color Enter number of pretraining epochs (i. e. from scratch)"
echo "Press Enter to use default [5] or type a number:$reset"
read -p "Pretraining epochs [5]: " pretrain_epochs
pretrain_epochs=${pretrain_epochs:-5}

echo "$color Enter number of fine-tuning epochs (i. e. from a trained model)"
echo "Press Enter to use default [5] or type a number:$reset"
read -p "Fine-tuning epochs [5]: " finetune_epochs
finetune_epochs=${finetune_epochs:-5}

# Update training commands with chosen parameters
T1_CMD="$T1_CMD_BASE --architecture $ARCHITECTURE -e 0 -z $finetune_epochs ./images ./trained_pretrained"
T2_CMD="$T2_CMD_BASE --architecture $ARCHITECTURE -e $pretrain_epochs -z 0 ./images ./trained_random"

# Update prefixes with both GPU and time profiling settings
update_prefixes "$gpu_index" "$use_time"

# Set the prefix based on the choice
case $choice in
    1|"")
        prefix=$LOCAL_PREFIX
        ;;
    2)
        prefix=$DOCKER_PREFIX
        ;;
    3)
        prefix=$SING_PREFIX
        $SING_PULL
        ;;
    *)
        echo "Invalid choice. Please run the script again and select 1, 2, or 3."
        exit 1
        ;;
esac

# Detect available CPU cores
AVAILABLE_CORES=$(nproc)
if [ $AVAILABLE_CORES -eq 1 ]; then
    DEFAULT_CORES=1
else
    DEFAULT_CORES=$((AVAILABLE_CORES - 1))
fi

echo "$color How many CPU cores would you like to use for computing?"
echo "System has $AVAILABLE_CORES cores available"
echo "Press Enter to use recommended default [$DEFAULT_CORES] or type a number:$reset"
read -p "Enter number of cores [$DEFAULT_CORES]: " NCORES
NCORES=${NCORES:-$DEFAULT_CORES}

# Rest of the script (image generation, training, query commands)
echo "$color$prefix $IM_CMD -n $NCORES$reset"
prepend_text IM "$prefix $IM_CMD -n $NCORES"

echo -e "$color$prefix $T1_CMD -n $NCORES$reset"
prepend_text T1 "$prefix $T1_CMD -n $NCORES"

echo -e "$color$prefix $T2_CMD -n $NCORES$reset"
prepend_text T2 "$prefix $T2_CMD -n $NCORES"

# Check if trained_pretrained/input_data.csv exists before running the loop
if [ -f "trained_pretrained/input_data.csv" ]; then
    # Create fastq_query directory if it doesn't exist
    mkdir -p fastq_query
    
    while IFS=, read -r sample bp kmer_mapping kmer_size path labels pos_qual is_valid; do
        if [ "$is_valid" = "True" ]; then
            # Find the source directory
            source_dir=$(find "./Bembidion" -type d -name "$sample" | head -n 1)
            if [ -n "$source_dir" ]; then
                # Create relative symlink in fastq_query directory
                (cd fastq_query && ln -sf "../$source_dir" "$sample")
            fi
        fi
    done < trained_pretrained/input_data.csv
else
    echo "${color}Warning: trained_pretrained/input_data.csv not found. Skipping query folder creation.$reset"
fi

if [ -d "fastq_query" ] && [ -f "trained_pretrained/trained_model.pkl" ]; then
    echo -e "$color$prefix $Q1_CMD -n $NCORES$reset"
    prepend_text Q1 "$prefix $Q1_CMD -n $NCORES"
else
    echo "${color}Warning: Required files or directories for Q1 command not found. Skipping.$reset"
fi

if [ -d "inferences_Bembidion/query_images" ]; then
    echo -e "$color$prefix $Q2_CMD -n $NCORES$reset"
    prepend_text Q2 "$prefix $Q2_CMD -n $NCORES"
else
    echo "${color}Warning: Required directory for Q2 command not found. Skipping.$reset"
fi

echo "${color}ALL TESTS CONCLUDED$reset"
echo "${color}If you want to remove files generated, use this command:$reset"
echo "${color}rm -rf varKoder.sif Bembidion fastq_query images inferences* trained* stats.csv$reset"
