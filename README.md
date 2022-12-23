# varKoder

A tool that uses **var**iation in **K**-mer frequecies as DNA barc**ode**s.

This python script can generate varKodes from raw reads, which are images encoding the relative frequecies of different k-mers in a genome. It can also train a convolutional neural network to recognize species based on these images, and query new samples using a trained model.

For more information see the publication:

``` Add paper here later ```

## Installing dependencies

varKoder is a python script but uses a few external programs to prepare varKodes:
- [fastp](https://github.com/OpenGene/fastp)
- [bbtools](https://jgi.doe.gov/data-and-tools/software-tools/bbtools/)
- [dsk](https://github.com/GATB/dsk)
- [pigz](https://zlib.net/pigz/)

Here we provide installation instructions for Linux and OSX, the only systems in which the program has been tested:

### Linux

All dependencies can be installed with [Anaconda](https://anaconda.org). For convenience, we provide a conda environment file with package versions that are compatible with the current version of the script. 
To install these dependencies as a new conda environment named `varKoder`, clone this github repository and use anaconda:

```console
git clone https://github.com/brunoasm/varKoder
cd varkoder
conda env create --file conda_environments/linux.yml
```
After installing the environment, you will be able to activate the environment by using:
```console
conda activate varKoder
```
### Mac

We tested this program using a mac with an ARM processor. Not all dependencies are available using anaconda, and for that reason the setup takes a few more steps. To start, create an Anaconda environment with the programs that are available:
```console
git clone https://github.com/brunoasm/varKoder
cd varkoder
conda env create --file conda_environments/mac.yml
```

To install `bbtools` and `fastp`, the easiest way is to use [Homebrew](https://brew.sh). Install homebrew and then run the following:
```console
brew tap brewsci/bio
brew install bbtools
brew install fastp
```

In macs with ARM processors, you may get an error when installing `fastp`. To workaround the error, you have to install [Rosetta2](https://support.apple.com/en-us/HT211861) so your mac can run intel-based programs, and additionally install the intel version of homebrew. You can accomplish this with the following:
```console
## install rosetta to run intel-based programs
/usr/sbin/softwareupdate --install-rosetta

## Install intel-version of homebrew
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

## install fastp with hombrew
arch -x86_64 /usr/local/Homebrew/bin/brew install fastp
```

Finally, `dsk` can be obtained as a binary executable and installed to your conda environment:
```
conda activate varKoder
wget https://github.com/GATB/dsk/releases/download/v2.3.3/dsk-v2.3.3-bin-Darwin.tar.gz
tar -xvzf dsk-v2.3.3-bin-Darwin.tar.gz
cd dsk-v2.3.3-bin-Darwin
cp bin/* $CONDA_PREFIX/bin/
```

After doing this, you can safely delete the downloaded file and folder `dsk-v2.3.3-bin-Darwin` and you needed to restart your terminal to be able to use `dsk`






### Installing

A step by step series of examples that tell you how to get a development environment running.

First, clone the repository to your local machine:


Next, navigate to the project directory and install the necessary dependencies:


Finally, run the program:


End with an example of getting some data out of the system or using it for a little demo.

## Running the tests

Explain how to run the automated tests for this system.

## Built With

* [Python](https://www.python.org/) - The programming language used
* [BioPython](https://biopython.org/) - A library for working with biological data in Python

## Associated Paper

* **Title of Paper** - [link to paper](link to paper)

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Your Name** - *Initial work* - [YourGitHubUsername](https://github.com/YourGitHubUsername)

See also the list of [contributors](https://github.com/YourGitHubUsername/project-name/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc.
