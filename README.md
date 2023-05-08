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

  * Multi-label (default): by default, varKoder uses [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification), in which each varKode can be associated with one or more labels. When querying an unknown sample, the response may be one or more of the labels used in training, or none, when the sample is not recognized as anything the model has been trained with. We found that this strategy is particularly useful to train a computer vision model to recognize when a sample was sequenced from low-quality DNA, preventing errors in which low-quality samples are predicted to be similar because of similarity in contamination and DNA damage. For this reason, varKoder automatically adds a flag to potentially low-quality samples when preparing varKodes. Another advantage of multi-label classification is the possibility of adding multiple taxonomic labels for each sample. This enables, for example, correct prediction of a higher taxonomic level even if a lower level (for example, species) is uncertain.
  
  * Single-label: in this strategy, each varKode is associated with a single label. Instead of predicting the confidence in each label independently, *varKoder* will output which of the labels used in training is the best one for a given query sample. This may be more straightforward to handle, since there will always be a response. But we found it to be more prone to errors. Evaluating the confidence in a particular prediction is also less straigthforward.

See options below in [Usage](#Usage) section on how to implement each strategy.

## Installing dependencies

Python packages required to run `varKoder` can be installed with anaconda (see instructions below for Linux and Mac).

It also uses a few external programs:
 - [fastp](https://github.com/OpenGene/fastp)
 - [bbtools](https://jgi.doe.gov/data-and-tools/software-tools/bbtools/)
 - [dsk](https://github.com/GATB/dsk)
 - [pigz](https://zlib.net/pigz/)

Here we provide installation instructions for Linux and OSX, the only systems in which the program has been tested:

### Linux

All dependencies can be installed with [Anaconda](https://anaconda.org). For convenience, we provide a conda environment file with package versions that are compatible with the current version of the script. 
To install these dependencies as a new conda environment named `varKoder`, clone this github repository and use anaconda:

```bash
git clone https://github.com/brunoasm/varKoder
cd varkoder
conda env create --file conda_environments/linux.yml
```
After installing the environment, you will be able to activate the environment by using:
```bash
conda activate varKoder
```

Now you are ready to use *varKoder*.

### Mac

We tested this program using a Mac laptop with an M2 processor. Not all dependencies are available using Anaconda, and for that reason the setup takes a few more steps. To start, create an Anaconda environment with the programs that are available:
```bash
git clone https://github.com/brunoasm/varKoder
cd varkoder
conda env create --file conda_environments/mac.yml
```

To install `fastp`, the easiest way is to use [Homebrew](https://brew.sh). Install homebrew and then run the following:
```bash
brew tap brewsci/bio
brew install fastp
```

In macs with ARM processors (i. e. M1 or M2), you may get an error when installing `fastp`, which was compiled for Intel. To workaround the error, you have to install [Rosetta2](https://support.apple.com/en-us/HT211861) so your Mac can run Intel-based programs. You will also need to  install the Intel version of homebrew. You can accomplish this with the following commands:

```bash
## install rosetta to run intel-based programs
/usr/sbin/softwareupdate --install-rosetta

## Install intel-version of homebrew
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

## install fastp with hombrew
arch -x86_64 /usr/local/Homebrew/bin/brew install fastp
```

Finally, `dsk` can be obtained as a binary executable and installed to your conda environment:
```bash
conda activate varKoder
wget https://github.com/GATB/dsk/releases/download/v2.3.3/dsk-v2.3.3-bin-Darwin.tar.gz
tar -xvzf dsk-v2.3.3-bin-Darwin.tar.gz
cd dsk-v2.3.3-bin-Darwin
cp bin/* $CONDA_PREFIX/bin/
```

After doing this, you can safely delete the downloaded file and folder `dsk-v2.3.3-bin-Darwin` and you need to restart your terminal to be able to use `dsk`


## Usage

varKoder is run as a python script. For example, assuming that the script is in a folder named `varKoder`, you can the program help by using:

```bash
python varKoder/varKoder.py -h
```

There are three commands available (`image`, `train` and `query`) and you can also get help on each command by using `-h`:
```bash
python varKoder/varKoder.py image -h
python varKoder/varKoder.py train -h
python varKoder/varKoder.py query -h
```

Follow these links for detailed information for each command:

1. [Creating varKodes with `varKoder.py image`](docs/image.md)
2. [Training an image classification model `varKoder.py train`](docs/train.md)
3. [Identifying and unknown sample with `varKoder.py query`](docs/query.md)


## Author

B. de Medeiros (Field Museum of Natural History), starting in 2019.


