#draft script to run varKoder tests
#TO DO: add varKoder installation, fast-dump to get data

set -e

#Testing locally installed varKoder
IM_CMD="varKoder image -k 7 -n 8 -c 2 -m 500K -M 20M -o ./images -f ./stats.csv ./input/files_corrected.csv"
T1_CMD="varKoder train -e 5 -f 5 --overwrite images trained"
T1_CMD="varKoder train --pretrained-timm -e 5 -f 5 --overwrite images trained"
Q1_CMD="varKoder query --overwrite -k 7 -n 10 -c 2 -M 20M --keep-images --model trained/trained_model.pkl fastq_query/ inferences"
Q2_CMD="varKoder query --overwrite -k 7 -n 10 -c 2 -M 20M -I query_images inferences_SRA"
SINGULARITY_PULL="singularity pull --docker-login ~/singularity/varKoder.sif docker://brunoasm/varkoder"
SINGULARITY_PREFIX="singularity exec --no-home --cleanenv --nv  -B $(pwd):/home /data/bdemedeiros/singularity/varKoder.sif"

$IM_CMD
$T1_CMD
$Q1_CMD
$Q2_CMD

$SINGULARITY_PULL

$SINGULARITY_PREFIX $IM_CMD
$SINGULARITY_PREFIX $T1_CMD
$SINGULARITY_PREFIX $Q1_CMD
$SINGULARITY_PREFIX $Q2_CMD
