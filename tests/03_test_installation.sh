source 02_constants.sh

# Test local installation with image, train and query commands
echo "$color$IM_CMD$reset"
prepend_text IM $IM_CMD

echo -e "$color$T1_CMD$reset"
prepend_text T1 $T1_CMD

echo -e "$color$T2_CMD$reset"
prepend_text T2 $T2_CMD

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

echo -e "$color$Q1_CMD$reset"
prepend_text Q1 $Q1_CMD

echo -e "$color$Q2_CMD$reset"
prepend_text Q2 $Q2_CMD

echo "${color}ALL TESTS CONCLUDED$reset"
echo "${color}If you want to remove files generated, use this command:$reset"
echo "${color}rm -rf Bembidion fastq_query images inferences* trained* stats.csv$reset"

