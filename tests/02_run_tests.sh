#draft script to run varKoder tests

set -e

TESTDIR=Bembidion

prepend_text() {
    local prefix="$1"
    shift
    "$@" | while read line; do
        echo "$prefix: $line"
    done
}


#Testing locally installed varKoder
IM_CMD="varKoder image --seed 1 -k 7 -n 8 -c 2 -m 500K -M 20M -o ./images $TESTDIR"
T1_CMD="varKoder train --seed 2 -e 5 -z 5 ./images ./trained"
T2_CMD="varKoder train --seed 3 --random-weigths -e 5 -z 5 --overwrite ./images ./trained"
Q1_CMD="varKoder query --seed 4 --overwrite -k 7 -n 20 -c 2 -M 20M --keep-images --model trained/trained_model.pkl fastq_query/ inferences"
Q2_CMD="varKoder query --seed 5 --overwrite -k 7 -n 10 -c 2 -M 20M -I inferences/query_images inferences_SRA"
SINGULARITY_PULL="singularity pull --force --docker-login ~/singularity/varKoder.sif docker://brunoasm/varkoder"
SINGULARITY_PREFIX="singularity exec --no-home --cleanenv --nv -B $TMPDIR:/tmp -B $PWD:/home --pwd /home /data/bdemedeiros/singularity/varKoder.sif"
DOCKER_PREFIX="docker run -v $TMPDIR:/tmp -v $PWD:/home brunoasm/varkoder:latest" 

prepend_text IM: $IM_CMD
prepend_text T1: $T1_CMD
prepend_text T2: $T2_CMD
#create a query folder with valid samples
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
done < trained/input_data.csv


prepend_text Q1: $Q1_CMD
prepend_text Q2: $Q2_CMD

#$SINGULARITY_PULL
#$SINGULARITY_PREFIX $IM_CMD
#$SINGULARITY_PREFIX $T1_CMD
#$SINGULARITY_PREFIX $T2_CMD
#$SINGULARITY_PREFIX $Q1_CMD
#$SINGULARITY_PREFIX $Q2_CMD

echo DONE

rm -rf fastq_query images inferences* trained* stats.csv

