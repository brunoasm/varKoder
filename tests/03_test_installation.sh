#!/bin/bash
# 03_test_installation.sh
source 02_constants.sh

# Initialize arrays to track successful commands and profiling data
successful_commands=()

# Check if bash version supports associative arrays
if [[ "${BASH_VERSINFO[0]}" -ge 4 ]]; then
    declare -A command_exit_codes
    declare -A command_wall_times
    declare -A command_cpu_times
    declare -A command_memory_usage
    USE_ASSOC_ARRAYS=true
else
    # Fallback for older bash versions (using indexed arrays)
    command_exit_codes=()
    command_wall_times=()
    command_cpu_times=()
    command_memory_usage=()
    command_names=()
    USE_ASSOC_ARRAYS=false
fi

# Function to run a command and capture its results
run_command() {
    local cmd_name="$1"
    local full_cmd="$2"
    
    echo "$color$full_cmd$reset"
    
    # Run the command and capture its exit code
    if [[ "$use_time" == "Y" || "$use_time" == "y" ]]; then
        # For profiled commands, capture the stderr separately to parse profiling info
        time_output_file=$(mktemp)
        eval "$full_cmd" 2> >(tee "$time_output_file" | while read -r line; do echo "$cmd_name: $line"; done)
        exit_code=${PIPESTATUS[0]}
        
        # Extract profiling data
        if [ -f "$time_output_file" ]; then
            wall_time=$(grep "Elapsed (wall clock) time" "$time_output_file" | sed 's/.*: //')
            cpu_time=$(grep "User time" "$time_output_file" | sed 's/.*: //')
            memory=$(grep "Maximum resident set size" "$time_output_file" | sed 's/.*: //')
            
            # Store command-specific profiling data in arrays using the appropriate method
            if [ "$USE_ASSOC_ARRAYS" = true ]; then
                # Use associative arrays
                command_wall_times[$cmd_name]="$wall_time"
                command_cpu_times[$cmd_name]="$cpu_time"
                command_memory_usage[$cmd_name]="$memory"
            else
                # Use indexed arrays with command names as reference
                command_names+=("$cmd_name")
                command_wall_times+=("$wall_time")
                command_cpu_times+=("$cpu_time")
                command_memory_usage+=("$memory")
            fi
            
            rm "$time_output_file"
        fi
    else
        # For non-profiled commands, just run with prepend_text for stderr
        prepend_text "$cmd_name" "$full_cmd"
        exit_code=$?
    fi
    
    # Record success/failure
    if [ "$USE_ASSOC_ARRAYS" = true ]; then
        command_exit_codes["$cmd_name"]=$exit_code
    else
        # For non-associative arrays, add the exit code to the end
        # If cmd_name already exists, find its index and update
        found=false
        for i in "${!command_names[@]}"; do
            if [[ "${command_names[$i]}" == "$cmd_name" ]]; then
                command_exit_codes[$i]=$exit_code
                found=true
                break
            fi
        done
        # If not found, add it
        if [ "$found" = false ]; then
            command_names+=("$cmd_name")
            command_exit_codes+=($exit_code)
        fi
    fi
    
    if [ $exit_code -eq 0 ]; then
        successful_commands+=("$cmd_name")
    fi
    
    return $exit_code
}

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

# Detect available CPU cores based on the operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS detection method
    AVAILABLE_CORES=$(sysctl -n hw.ncpu)
else
    # Linux detection method
    AVAILABLE_CORES=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 1)
fi

# Handle the case when detection fails
if [[ -z "$AVAILABLE_CORES" || "$AVAILABLE_CORES" -lt 1 ]]; then
    AVAILABLE_CORES=1
fi

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

# Create FASTA test data before running image tests
create_fasta_test_data

# Run commands and track success/failure
# Test with FASTQ files (original test)
echo "${color}Testing image generation with FASTQ files...$reset"
run_command "IM" "$prefix $IM_CMD -n $NCORES"

# Test with FASTA files (new test with 1 Kbp minimum)
if [ -f "${TESTDIR_FASTA}_input.csv" ]; then
    echo "${color}Testing image generation with FASTA files (1 Kbp minimum)...$reset"
    run_command "IM_FASTA" "$prefix $IM_FASTA_CMD -n $NCORES"
else
    echo "${color}Warning: ${TESTDIR_FASTA}_input.csv not found. Skipping FASTA image test.$reset"
fi
run_command "C" "$prefix $C_CMD -n $NCORES"
run_command "T1" "$prefix $T1_CMD -n $NCORES"
run_command "T2" "$prefix $T2_CMD -n $NCORES"

# Check if trained_pretrained/input_data.csv exists before running the loop
if [ -f "trained_pretrained/input_data.csv" ]; then
    # Create fastq_query directory if it doesn't exist
    mkdir -p fastq_query
    
    while IFS=, read -r sample bp kmer_mapping kmer_size path labels pos_qual is_valid; do
        if [ "$is_valid" = "True" ]; then
            # Find the source directory
            source_dir=$(find "./Bembidion" -type d -name "$sample" | head -n 1)
            if [ -n "$source_dir" ]; then
                # Remove ./ prefix from source_dir if present for clean symlink paths
                clean_source_dir="${source_dir#./}"
                # Create relative symlink in fastq_query directory
                (cd fastq_query && ln -sf "../$clean_source_dir" "$sample")
            fi
        fi
    done < trained_pretrained/input_data.csv
else
    echo "${color}Warning: trained_pretrained/input_data.csv not found. Skipping query folder creation.$reset"
fi

if [ -d "fastq_query" ] && [ -f "trained_pretrained/trained_model.pkl" ]; then
    run_command "Q1" "$prefix $Q1_CMD -n $NCORES"
else
    echo "${color}Warning: Required files or directories for Q1 command not found. Skipping.$reset"
fi

if [ -d "inferences_Bembidion/query_images" ]; then
    run_command "Q2" "$prefix $Q2_CMD -n $NCORES"
else
    echo "${color}Warning: Required directory for Q2 command not found. Skipping.$reset"
fi

# Test query command with FASTA images using default model
if [ -d "images_fasta" ] && [ -n "$(find images_fasta -name '*.png' 2>/dev/null)" ]; then
    echo "${color}Testing query with FASTA images using default model...$reset"
    run_command "Q_FASTA" "$prefix $Q_FASTA_CMD -n $NCORES"
else
    echo "${color}Warning: FASTA images directory not found or empty. Skipping FASTA query test.$reset"
fi

# Print summary
echo -e "\n${color}===== TEST SUMMARY =====$reset"
echo -n "${color}Mode: "
case $choice in
    1|"") echo "local conda";;
    2) echo "docker";;
    3) echo "singularity";;
    *) echo "unknown";;
esac
echo "$reset"
echo "${color}Architecture: $ARCHITECTURE$reset"
echo "${color}CPU Cores: $NCORES$reset"
if [ "$IS_MAC_ARM" = false ] && [ -n "$gpu_index" ]; then
    echo "${color}GPU: $gpu_index$reset"
else
    echo "${color}GPU: None (CPU mode)$reset"
fi

echo -e "\n${color}Commands completed successfully:$reset"
if [ ${#successful_commands[@]} -eq 0 ]; then
    echo "${color}None$reset"
else
    for cmd in "${successful_commands[@]}"; do
        echo -n "${color}- $cmd$reset"
        # Display command-specific details
        case $cmd in
            "IM") echo "${color} (Image generation from FASTQ)$reset" ;;
            "IM_FASTA") echo "${color} (Image generation from FASTA, 1 Kbp min)$reset" ;;
            "C") echo "${color} (Image conversion)$reset" ;;
            "T1") echo "${color} (Training with pre-trained weights, $finetune_epochs epochs)$reset" ;;
            "T2") echo "${color} (Training from scratch, $pretrain_epochs epochs)$reset" ;;
            "Q1") echo "${color} (Query from FASTQ)$reset" ;;
            "Q2") echo "${color} (Query from images)$reset" ;;
            "Q_FASTA") echo "${color} (Query from FASTA images with default model)$reset" ;;
            *) echo "" ;;
        esac
    done
fi

echo -e "\n${color}Commands that failed:$reset"
failed=false
for cmd in IM IM_FASTA C T1 T2 Q1 Q2 Q_FASTA; do
    if [[ " ${successful_commands[*]} " != *" $cmd "* ]]; then
        # Get the exit code using the appropriate array type
        exit_code=""
        if [ "$USE_ASSOC_ARRAYS" = true ]; then
            exit_code="${command_exit_codes[$cmd]}"
        else
            for i in "${!command_names[@]}"; do
                if [[ "${command_names[$i]}" == "$cmd" ]]; then
                    exit_code="${command_exit_codes[$i]}"
                    break
                fi
            done
        fi
        
        # Only display if we have an exit code (command was actually run)
        if [ -n "$exit_code" ]; then
            echo "${color}- $cmd (Exit code: $exit_code)$reset"
            failed=true
        fi
    fi
done
if [ "$failed" = false ]; then
    echo "${color}None$reset"
fi

# Display profiling information if it was enabled
if [[ "$use_time" == "Y" || "$use_time" == "y" ]]; then
    echo -e "\n${color}Resource usage (from /usr/bin/time):$reset"
    printf "${color}%-5s %-20s %-20s %-30s\n$reset" "CMD" "Wall Time" "CPU Time" "Max RSS"
    echo "${color}-------------------------------------------------------------------$reset"
    
    # Loop through commands with the appropriate array method
    for cmd in IM IM_FASTA C T1 T2 Q1 Q2 Q_FASTA; do
        wall_time=""
        cpu_time=""
        memory=""
        
        if [ "$USE_ASSOC_ARRAYS" = true ]; then
            # Use associative arrays
            wall_time="${command_wall_times[$cmd]}"
            cpu_time="${command_cpu_times[$cmd]}"
            memory="${command_memory_usage[$cmd]}"
        else
            # Find the command in the indexed arrays
            for i in "${!command_names[@]}"; do
                if [[ "${command_names[$i]}" == "$cmd" ]]; then
                    wall_time="${command_wall_times[$i]}"
                    cpu_time="${command_cpu_times[$i]}"
                    memory="${command_memory_usage[$i]}"
                    break
                fi
            done
        fi
        
        if [ -n "$wall_time" ]; then
            printf "${color}%-5s %-20s %-20s %-30s\n$reset" \
                "$cmd" "$wall_time" "$cpu_time" "$memory"
        fi
    done
fi

echo -e "\n${color}ALL TESTS CONCLUDED$reset"
echo "${color}If you want to remove files generated, use this command:$reset"
echo "${color}rm -rf Bembidion_fasta Bembidion_fasta_input.csv fastq_query images images_fasta images_varkode inferences* trained* stats.csv Bembidion$reset"
