#draft script to run varKoder tests

set -e

TESTDIR=Bembidion

prepend_text() {
    local prefix="$1"
    shift
    "$@" 2> >(while read line; do echo "$prefix: $line"; done)
}

color=$(tput setaf 1)
reset=$(tput sgr0)


IM_CMD="varKoder image --seed 1 -k 7 -n 8 -c 2 -m 500K -M 20M -o ./images $TESTDIR"
T1_CMD="varKoder train --seed 2 -e 5 -z 5 ./images ./trained_pretrained"
T2_CMD="varKoder train --seed 3 --random-weigths -e 5 -z 5 --overwrite ./images ./trained_random"
Q1_CMD="varKoder query --seed 4 --overwrite -k 7 -n 20 -c 2 -M 20M --keep-images --model trained_pretrained/trained_model.pkl fastq_query/ inferences_Bembidion"
Q2_CMD="varKoder query --seed 5 --overwrite -k 7 -n 10 -c 2 -M 20M -I inferences_Bembidion/query_images inferences_SRA"
SINGULARITY_PULL="singularity pull --force --docker-login ~/singularity/varKoder.sif docker://brunoasm/varkoder"
SINGULARITY_PREFIX="singularity exec --no-home --cleanenv --nv -B $TMPDIR:/tmp -B $PWD:/home --pwd /home /data/bdemedeiros/singularity/varKoder.sif"
DOCKER_PREFIX="docker run -v $TMPDIR:/tmp -v $PWD:/home brunoasm/varkoder:latest" 

