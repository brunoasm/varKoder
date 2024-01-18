source 02_constants.sh

echo "$color What would you like to test? Please choose an option by typing the number:"
echo "1. installation using conda and pip"
echo "2. docker image"
echo "3. singularity image$reset"

# Read user input
read -p "Enter your choice (1/2/3): " choice

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

# Test local installation with image, train and query commands
echo "$color$prefix $IM_CMD$reset"
prepend_text IM $prefix $IM_CMD

echo -e "$color$prefix $T1_CMD$reset"
prepend_text T1 $prefix $T1_CMD

echo -e "$color$prefix $T2_CMD$reset"
prepend_text T2 $prefix $T2_CMD

#create a query folder with validation samples
while IFS=, read -r _ sample _ _ _ _ is_valid; do
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

echo -e "$color$prefix $Q1_CMD$reset"
prepend_text Q1 $prefix $Q1_CMD

echo -e "$color$prefix $Q2_CMD$reset"
prepend_text Q2 $prefix $Q2_CMD

echo "${color}ALL TESTS CONCLUDED$reset"
echo "${color}If you want to remove files generated, use this command:$reset"
echo "${color}rm -rf varKoder.sif Bembidion fastq_query images inferences* trained* stats.csv$reset"

