#varKoder test constants

TESTDIR=Bembidion

prepend_text() {
    local prefix="$1"
    shift
    "$@" 2> >(while read line; do echo "$prefix: $line"; done)
}

color=$(tput setaf 1)
reset=$(tput sgr0)

# Check if TMPDIR is not set
if [ -z "$TMPDIR" ]; then
    # TMPDIR is not set, so create a temporary directory and assign it
    TMPDIR=$(dirname $(mktemp -u -t tmp.XXXXXXXXXX))
fi

IM_CMD="image --seed 1 -k 7 -c 1 -m 500K -M 20M -o ./images $TESTDIR"
T1_CMD="train --seed 2 -e 0 -z 5 ./images ./trained_pretrained"
T2_CMD="train --architecture resnet18 --seed 3 --random-weights -e 5 -z 5 --overwrite ./images ./trained_random"
Q1_CMD="query --include-probs --seed 4 --overwrite -k 7 -c 1 -M 20M --keep-images --model trained_pretrained/trained_model.pkl fastq_query/ inferences_Bembidion"
Q2_CMD="query --threshold 0.5 --seed 5 --overwrite -k 7 -c 1 -M 20M -I inferences_Bembidion/query_images inferences_SRA"

SING_PULL="singularity pull --force varKoder.sif docker://brunoasm/varkoder"

LOCAL_PREFIX="varKoder"
SING_PREFIX="singularity exec --no-home --cleanenv --nv -B ${TMPDIR}:/tmp -B ${PWD}:/home --pwd /home varKoder.sif varKoder"
DOCKER_PREFIX="docker run --platform linux/amd64 -v $TMPDIR:/tmp -v $PWD:/home brunoasm/varkoder:latest" 

