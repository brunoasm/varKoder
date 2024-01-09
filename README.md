# varKoder

A tool that uses **var**iation in **K**-mer frequecies as DNA barc**ode**s.

This python script can generate varKodes from raw reads, which are images encoding the relative frequecies of different k-mers in a genome. It can also train a convolutional neural network to recognize species based on these images, and query new samples using a trained model.

For more information see the publication:

``` Add paper here later ```

## Approach

With *varKoder*, we use very low coverage whole genome sequencing to produce images that represent the genome composition of a sample. These images look like this, for example:

``` Add examples ```

We then use well-established image classification models to train a neural network using these images so it can learn to associate *varKodes* with labels associated with them. Often, these labels will be the known taxonomic identification of a sample, such as its species or genus. However, our approach is very general and the labels could include any other features of interest.

Finally, you can use *varKoder* to predict labels for an unknown sample, starting from sequencing reads or from pre-produced *varKode* images.

There are two possible strategies for image classification using *varKoder*:

  * Multi-label (default): by default, varKoder uses [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification), in which each varKode can be associated with one or more labels. When querying an unknown sample, the response may be one or more of the labels used in training, or none, when the sample is not recognized as anything the model has been trained with. We found that this strategy is particularly useful to train a computer vision model to recognize when a sample was sequenced from low-quality DNA, preventing errors in which low-quality samples are predicted to be similar because of similarity due to contamination and/or DNA damage. For this reason, varKoder automatically adds a flag to potentially low-quality samples when preparing varKodes. Another advantage of multi-label classification is the possibility of adding multiple taxonomic labels for each sample. This enables, for example, correct prediction of a higher taxonomic level even if a lower level (for example, species) is uncertain.
  
  * Single-label: in this strategy, each varKode is associated with a single label. Instead of predicting the confidence in each label independently, *varKoder* will output which of the labels used in training is the best one for a given query sample. This may be more straightforward to handle, since there will always be a response. But we found it to be more prone to errors. Evaluating the confidence in a particular prediction is also less straigthforward.

See options below in [Usage](#Usage) section on how to implement each strategy.

## Installation

To install varKoder, it is best to use Anaconda (or Anaconda and homebrew, for macs) to install dependencies, and then pip to install varKoder itself to your conda environment. Follow the instructions below to accomplish that.

Python packages required to run `varKoder` can be installed with anaconda (see instructions below for Linux and Mac).

It also uses a few external programs:
 - [fastp](https://github.com/OpenGene/fastp)
 - [bbtools](https://jgi.doe.gov/data-and-tools/software-tools/bbtools/)
 - [dsk](https://github.com/GATB/dsk)
 - [pigz](https://zlib.net/pigz/)

Here we provide installation instructions for Linux and OSX, the only systems in which the program has been tested:

### Linux

All dependencies can be installed with [conda](https://anaconda.org). For convenience, we provide a conda environment file with package versions that are compatible with the current version of the progam. 

To install dependencies and varKoder as a new conda environment named `varKoder`, use these commands

```bash
git clone https://github.com/brunoasm/varKoder
cd varKoder
conda env create --file conda_environments/linux.yml
conda activate varKoder
pip install .
```

If this takes too long, you can try using [mamba](https://github.com/mamba-org/mamba) instead, which should be much faster than conda. Follow instructions to install mamba and use the same command as above, but replacing `conda` with `mamba`.

### Mac

We tested this program using Macs with ARM processors (M1,M2,etc). Not all dependencies are available using Anaconda, and for that reason the setup takes a few more steps. To start, create an Anaconda environment with the programs that are available through conda and install varKoder to the conda environment:
```bash
git clone https://github.com/brunoasm/varKoder
cd varkoder
conda env create --file conda_environments/mac.yml
conda activate varKoder
pip install .
```

Currently, `dsk` for macs is not available through Anaconda. It can be obtained as a binary executable and installed to your conda environment. This code will download, install and remove the installer:
```bash
conda activate varKoder
wget https://github.com/GATB/dsk/releases/download/v2.3.3/dsk-v2.3.3-bin-Darwin.tar.gz
tar -xvzf dsk-v2.3.3-bin-Darwin.tar.gz
cd dsk-v2.3.3-bin-Darwin
cp bin/* $CONDA_PREFIX/bin/
cd ..
rm -r dsk-v2.3.3-bin-Darwin.tar.gz dsk-v2.3.3-bin-Darwin
```

The latest SRA toolkit is required to run tests and examples, but not for basic varKoder functionality.On Macs, it is better to install it with Homebrew. See instructions here: https://formulae.brew.sh/formula/sratoolkit 


## Usage

varKoder is run as a python script. For example, assuming that the script is in a folder named `varKoder`, you can the program help by using:

```bash
varKoder -h
```

There are three commands available (`image`, `train` and `query`) and you can also get help on each command by using `-h`:
```bash
varKoder image -h
varKoder train -h
varKoder query -h
```

Follow these links for detailed information for each command:

1. [Creating varKodes with `varKoder.py image`](docs/image.md)
2. [Training an image classification model `varKoder.py train`](docs/train.md)
3. [Identifying and unknown sample with `varKoder.py query`](docs/query.md)


## Author

B. de Medeiros (Field Museum of Natural History), starting in 2019.


