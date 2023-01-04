# varKoder

A tool that uses **var**iation in **K**-mer frequecies as DNA barc**ode**s.

This python script can generate varKodes from raw reads, which are images encoding the relative frequecies of different k-mers in a genome. It can also train a convolutional neural network to recognize species based on these images, and query new samples using a trained model.

For more information see the publication:

``` Add paper here later ```

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

We tested this program using a Mac Studio with an M1 processor. Not all dependencies are available using Anaconda, and for that reason the setup takes a few more steps. To start, create an Anaconda environment with the programs that are available:
```bash
git clone https://github.com/brunoasm/varKoder
cd varkoder
conda env create --file conda_environments/mac.yml
```

To install `bbtools` and `fastp`, the easiest way is to use [Homebrew](https://brew.sh). Install homebrew and then run the following:
```bash
brew tap brewsci/bio
brew install bbtools
brew install fastp
```

In macs with ARM processors (i. e. M1 or M2), you may get an error when installing `fastp`, which was compiled for Intel. To workaround the error, you have to install [Rosetta2](https://support.apple.com/en-us/HT211861) so your Mac can run Intel-based programs. You will also need to  install the intel version of homebrew. You can accomplish this with the following commands:

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

Below we provide detailed information for each command:

### varKoder.py image

This command processes raw sequencing reads and produces *varKodes*, which are images representing the variation in K-mer frequencies for a given k-mer size.

Processing includes the following steps:

 - Raw read cleaning: adapter removal, overlapping pair merging, exact duplicate removal.
 - Raw read subsampling: random subsampling of raw read files into files with fewer number of reads. This is useful in machine learning training as a data augmentation technique to make sure inferences are robust to random variations in sequencing and amount of input data.
 - Kmer counting: count of kmers from subsampled files.
 - VarKode geenration: generation of images from kmer counts, our *varKodes* that represent the genome composition of a taxon.

Optionally, the command can be used to preprocess sequences without creating images. 

#### Input format

`varKoder.py image` accepts two input formats: folder-based or table-based

##### folder input format

In this format, each taxonomic entity is represented by a folder, and within that folder there are subfolders for each sample. The latter contains all read files associated with a sample.

Higher-level folder names must correspond to the desired names, and subfolder names must correspond to sample IDs. Sequence file names have no constraint, other than explicitly marking if they are read 1 or 2 in case of paired reads. Any of the default naming conventions for paired reads should work, as long as the root name of paired files is the same.

Species and sample names cannot contain the following characters: `@` and `+`.

For example, let's assume that you have data for 2 species, and for each species you sequenced four samples. The files could be structured, for example, like this:

- sequence_data
   - Species_A
       - Sample_1234
          - 1234.R1.fq.gz
          - 1234.R2.fq.gz
          - 1234_run2_R1_.fq.gz
          - 1234_run2_R2_.fq.gz
       - Sample_1235
          - 1235_1.fq
          - 1235_2.fq
       - Sample_1236
          - sampleAA.fastq
       - Sample_1237
          - speciesA_seq1_R1.fq
          - speciesA_seq1_R2.fq
          - resequence.spA.1.fq.gz
          - resequence.spA.2.fq.gz
   - Species_cf._B
       - Sample_1300
          - anotherseq_R1_CTTAGTGC.fq
          - anotherseq_R2_CTTAGTGC.fq
       - Sample_1301
          - onemore1301_TGTGTCAA_1_A.fastq.gz
          - onemore1301_TGTGTCAA_2_A.fastq.gz
       - Sample_1302
          - 1302B_TTGTGTAGC_1.fastq
          - 1302B_TTGTGTAGC_2.fastq
       - Sample_1303
          - 1303.1.fastq.gz
          - 1303.2.fastq.gz
          - 1303.unpaired.fastq.gz
       
In the above example we show some possibilities for formatting the input. Ideally you would want to make it more organized, but we want to show that the input is robust to variation as long as the folder structure is respected:
 
- The first folder level indicates species name. It can contain special characters but not `@` or `+`.
- The second folder level groups all `fastq` files for a given sample. It can contain special characters but not `@` or `+`.
- Both compressed and uncompressed `fastq` files are accepted. Compressed files must end with the extension `.gz`. 
- Read pairs must have corresponding root file names and contain an indication of whether they are read 1 or 2. All examples above work for that end.
- Read files without `.1.`, `_1_`, `R1`, etc. will be considered unpaired.

##### table input format

In this case, you must provide a table in `csv` format linking each `fastq` file to its sample information.
The required column names are `taxon`,`sample` and `reads_file`. Just like with folder input format, taxon and sample names cannot use characters `@` or `+`.

This is an example of a csv file with the necessary information:

```csv
taxon,sample,reads_file
Acridocarpus,ugandensis_12,12_GGGAGCTAGTGG.R1.fastq.gz
Acridocarpus,ugandensis_12,12_GGGAGCTAGTGG.R2.fastq.gz
Acridocarpus,smeathmannii_168,168_TATGTCACATGG.R1.fastq.gz
Acridocarpus,smeathmannii_168,168_TATGTCACATGG.R2.fastq.gz
Acridocarpus,macrocalyx_176,176_GGACATGACCGG.R1.fastq.gz
Acridocarpus,macrocalyx_176,176_GGACATGACCGG.R2.fastq.gz
Acridocarpus,spectabilis_181,181_CTGTGATTTATT.R1.fastq.gz
Acridocarpus,spectabilis_181,181_CTGTGATTTATT.R2.fastq.gz
Acridocarpus,obovatus_3317,3317.R1.fq
Acridocarpus,obovatus_3317,3317.R2.fq
```

Note:
 - file paths must be given relative to the folder where the `csv` file is located (in the example above, they are all in the same folder).
 - each line represents one `fastq` file. Taxon and sample names can be repeated as needed.
 
#### Arguments

##### Required arguments

| argument | description |
| --- | --- |
| `input` | path to either the folder with fastq files or csv file relating file paths to samples. See input formats above. |

##### Optional arguments

| argument | description |
| --- | --- |
| `-h`, `--help` | show help message and exit. |
| `-d SEED`, `--seed SEED` |  optional random seed to make sample preprocessing reproducible. |
| `-x`, `--overwrite` | overwrite existing results. By default samples are skipped if files exist. |
| `-v`, `--verbose` |  show output for `fastp`, `dsk` and `bbtools`. By default these are ommited. This may be useful in debugging if you get errors. |
| `-k KMER_SIZE`, `--kmer-size KMER_SIZE` | size of kmers to count. Sizes from 5 to 9 are supported at the moment. (default: 7) |
| `-n N_THREADS`, `--n-threads N_THREADS` | number of samples to preprocess in parallel. See notes below on usage. (default: 1) |
| `-c CPUS_PER_THREAD`, `--cpus-per-thread CPUS_PER_THREAD` | number of cpus to use for preprocessing each sample. See notes below on usage (default: 1) |
| `-o OUTDIR`, `--outdir OUTDIR` | path to folder where to write final images. (default: images) |
| `-f STATS_FILE`, `--stats-file STATS_FILE`} | path to file where sample statistics will be saved. See *Output* below for details (default: stats.csv) |
| `-i INT_FOLDER`, `--int-folder INT_FOLDER` | folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used. See *Output* below for details. |
| `-m MIN_BP`, `--min-bp MIN_BP` | minimum number of post-cleaning basepairs to make an image. Samples below this threshold will be discarded (default: 10M) |
| `-M MAX_BP`, `--max-bp MAX_BP` | maximum number of post-cleaning basepairs to make an image. By default this is all the data available for each sample. |
| `-a`, `--no-adapter` |      do not attempt to remove adapters from raw reads. See notes below for details. |
| `-r`, `--no-merge` |        do not attempt to merge paired reads. See notes below for details. |
| `-X`, `--no-image` |       clean and split raw reads, but do not generate image. You must provide a folder to save intermediate files with `--int-folder` to keep the processed reads. |

Notes:

If `--n-threads` is more than 1, `varKoder` will use Python `multiprocessing` library to do sample preprocessing in parallel (i. e. clean, split raw reads and generate images). If `-cpus-per-thread` is more than one, the number of CPUs will be passed to programs (i. e. `fastp`, 'dsk', 'bbtools') when processing a sample: this is the number of cores dedicated to each sample. So a user can select to parallelize computing for each sample, or to do more than one sample in parallel, or both. We have not extensively tested the potential speed ups of each method.

If `--max-bp` is set, `varKoder` first trims raw read files to 5 times the value of `--max-bp`. This can speed up raw read cleaning and kmer counting, but it means that data beyond this number of basepairs in input files will be ignored.

By default, we use `fastp` to remove adapters and merge overlapping paired reads. Both can ve turned off with `--no-adapter` amd `--no-merge`, respectively. Exact duplicates are always removed with `bbtools`, currently this cannot be turned off.




#### Output

varKode Images produced will be saved to a folder named `images` (or another name is the `--outdir` argument is provided). Each image will be named with the following convention:

```taxon+sample@[thousands_of_bp]K+k[kmer_length].png```

For example:

```Acridocarpus+ugandensis_12@00010000K+k7.png``` is a varKode of the taxon `Acridocarpus`, sample id `ugandensis_12`, made from 10 million base pairs and for a kmer length of `7`.

It will also save a csv table with sample processing statistics. By default, this is `stats.csv` but it can be changed with the `--stats-file` argument.

By default, only the stats file and final images are saved. Intermediate files (clean reads, `fastp` reports, processed reads, kmer counts) are saved in a temporary folder and deleted when `varKoder` finishes processing. To save these files instead, provide a folder path with the `--int-folder` argument. In the provided path, `varKoder` will save 3 folders: 

    `clean_reads` fastq files adapter removed and merged [also trimmed if `--max-bp` provided])
    `split_fastqs` subsampled fastq files from clean reads
    `Xmer_counts` `dsk` kmer count files (with `X` being the kmer length)


### varKoder.py train

After *varKodes* are generated with `varKoder.py image`, they can be used to train a neural network to recognize taxa based on these images. The `varKoder.py train` command uses `fastai` and `pytorch` to do this training, with image models obtained with the `timm` library.

#### Arguments

##### Required arguments

`input` either a path to a folder (if using folder input format) or to a csv file (if using table input format)

##### Optional arguments



### varKoder.py query


## Authors

* **Your Name** - *Initial work* - [YourGitHubUsername](https://github.com/YourGitHubUsername)

See also the list of [contributors](https://github.com/YourGitHubUsername/project-name/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc.
