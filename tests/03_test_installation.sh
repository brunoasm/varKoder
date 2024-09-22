#!/bin/bash
source 02_constants.sh

echo "$color What would you like to test? Please choose an option by typing the number:"
echo "1. local conda installation"
echo "2. docker image"
echo "3. singularity image$reset"
# Read user input
read -p "Enter your choice (1/2/3): " choice

# GPU selection (skip for Mac ARM)
if [ "$IS_MAC_ARM" = false ]; then
    if [ -n "$AVAILABLE_GPUS" ]; then
        echo "$color Available GPUs: $AVAILABLE_GPUS"
        echo "Enter the index of the GPU you want to use, or press Enter to use CPU:$reset"
        read -p "GPU index: " gpu_index
        set_cuda_visible_devices "$gpu_index"
        update_prefixes "$gpu_index"
    else
        echo "$color No NVIDIA GPUs detected. Using CPU.$reset"
    fi
else
    echo "$color Mac ARM system detected. GPU selection is not applicable.$reset"
fi

# Set the prefix based on the choice
case $choice in
    1)
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

echo "$color How many cores you have available for computing? Type the number and hit enter:$reset"
read -p "Enter number of cores: " NCORES

# Test installation with image, train and query commands
echo "$color$prefix $IM_CMD -n $NCORES$reset"
prepend_text IM "$prefix $IM_CMD -n $NCORES"

echo -e "$color$prefix $T1_CMD -n $NCORES$reset"
prepend_text T1 "$prefix $T1_CMD -n $NCORES"

echo -e "$color$prefix $T2_CMD -n $NCORES$reset"
prepend_text T2 "$prefix $T2_CMD -n $NCORES"

# Check if trained_pretrained/input_data.csv exists before running the loop
if [ -f "trained_pretrained/input_data.csv" ]; then
    #create a query folder with validation samples
    while IFS=, read -r sample *   *is_valid; do
        if [ "$is_valid" = "True" ]; then
            mkdir -p fastq_query
            for dir in "./Bembidion/"*"/$sample"; do
                if [ -d "$dir" ]; then
                    cd fastq_query
                    ln -sf .$dir $sample
                    cd ..
                    break
                fi
            done
        fi
    done < trained_pretrained/input_data.csv
else
    echo "${color}Warning: trained_pretrained/input_data.csv not found. Skipping query folder creation.$reset"
fi

# Check if required directories exist before running query commands
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
