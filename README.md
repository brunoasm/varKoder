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

We tested this program using a Mac laptop with an M2 processor. Not all dependencies are available using Anaconda, and for that reason the setup takes a few more steps. To start, create an Anaconda environment with the programs that are available:
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
| `-v`, `--verbose` |  show output for `fastp`, `dsk` and `bbtools`. By default these are ommited. This may be useful in debugging if you get errors. |
| `-x`, `--overwrite` | overwrite existing results. By default samples are skipped if files exist. |
| `-k KMER_SIZE`, `--kmer-size KMER_SIZE` | size of kmers to count. Sizes from 5 to 9 are supported at the moment. (default: 7) |
| `-n N_THREADS`, `--n-threads N_THREADS` | number of samples to preprocess in parallel. See tips below on usage. (default: 1) |
| `-c CPUS_PER_THREAD`, `--cpus-per-thread CPUS_PER_THREAD` | number of cpus to use for preprocessing each sample. See tips below on usage (default: 1) |
| `-o OUTDIR`, `--outdir OUTDIR` | path to folder where to write final images. (default: images) |
| `-f STATS_FILE`, `--stats-file STATS_FILE`} | path to file where sample statistics will be saved. See *Output* below for details (default: stats.csv) |
| `-i INT_FOLDER`, `--int-folder INT_FOLDER` | folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used. See *Output* below for details. |
| `-m MIN_BP`, `--min-bp MIN_BP` | minimum number of post-cleaning basepairs to make an image. Samples below this threshold will be discarded. You can use SI abbreviations (e. g. 1M for 1 million or 150K for 150 thousand bp) (default: 10M) |
| `-M MAX_BP`, `--max-bp MAX_BP` | maximum number of post-cleaning basepairs to make an image. By default this is all the data available for each sample. You can use SI abbreviations (e. g. 1M for 1 million or 150K for 150 thousand bp) |
| `-a`, `--no-adapter` |      do not attempt to remove adapters from raw reads. See notes below for details. |
| `-r`, `--no-merge` |        do not attempt to merge paired reads. See notes below for details. |
| `-X`, `--no-image` |       clean and split raw reads, but do not generate image. You must provide a folder to save intermediate files with `--int-folder` to keep the processed reads. |

#### Image command tips

The defaults for optional arguments were chosen based on our sets. Here are some tips to help you choose values for optional arguments in case you want to change them:

 1. `--min-bp` and `--max-bp`. These arguments define how many subsampled images will be generated for each sample. We use several varKodes per sample with different amounts of input data so that the neural network can be trained to ignore random fluctuations in kmer counts and focus on features that define taxa. The rule used by `varKoder.py` is to create images for subsampled sequence files with standardized input ampounts of data, between `--min-bp` amd `--max-bp` with bp amounts corresponding to 1, 2, or 5 in each order of magnitude. For example, if `--min-bp` is 1M (1 million) and `--max-bp` is 100M (100 million), you will get 7 varKode images for each sample, corresponding to randomly chosen reads for input amounts 1M, 2M, 5M, 10M, 20M, 50M and 100M of base pairs. If `--max-bp` is ommitted, we will follow this run until we reach the largest input amount possible for the number of raw reads in each sample, including a varKode based on all of the data available.
 
 2. If `--max-bp` is set, `varKoder` first trims raw read files to 5 times the value of `--max-bp`. This can speed up raw read cleaning and kmer counting, but it means that subsampled files are randomly chosen from the reads available in this trimmed raw read files, not the whole initial raw reads.

 3. `--no-merge` and `--no-adpater`. We have not extensively tested the effect of skipping adapter trimming and merging of overlapping reads. It may be the case that these preprocessing steps are unnecessary in your case and you can speed up computation by using these arguments and skipping these steps.

 4. If `--n-threads` is more than 1, `varKoder` will use Python `multiprocessing` library to do sample preprocessing in parallel (i. e. clean, split raw reads and generate images). If `-cpus-per-thread` is more than one, the number of CPUs will be passed to programs (i. e. `fastp`, `dsk`, `bbtools`) when processing a sample: this is the number of cores dedicated to each sample. So a user can select to parallelize computing for each sample, or to do more than one sample in parallel, or both. We have not extensively tested the potential speed ups of each method.


#### Output

varKodes will be saved as png images to a folder named `images` (or another name is the `--outdir` argument is provided). Each image will be named with the following convention:

```taxon+sample@[thousands_of_bp]K+k[kmer_length].png```

For example:

```Acridocarpus+ugandensis_12@00010000K+k7.png``` is a varKode of the taxon `Acridocarpus`, sample id `ugandensis_12`, made from 10 million base pairs and for a kmer length of `7`.

It will also save a csv table with sample processing statistics. By default, this is `stats.csv` but it can be changed with the `--stats-file` argument.

By default, only the stats file and final images are saved. Intermediate files (clean reads, `fastp` reports, processed reads, kmer counts) are saved in a temporary folder and deleted when `varKoder` finishes processing. To save these files instead, provide a folder path with the `--int-folder` argument. In the provided path, `varKoder` will save 3 folders: 

 `clean_reads`: fastq files adapter removed and merged (also trimmed if `--max-bp` provided)
 `split_fastqs`: subsampled fastq files from clean reads
 `Xmer_counts`: `dsk` kmer count files (with `X` being the kmer length)


### varKoder.py train

After *varKodes* are generated with `varKoder.py image`, they can be used to train a neural network to recognize taxa based on these images. The `varKoder.py train` command uses `fastai` and `pytorch` to do this training, with image models obtained with the `timm` library. 

#### Arguments

##### Required arguments

| argument | description |
| --- | --- |
| input       |          path to the folder with input images. These must have been generated with `varKoder.py image` |
| outdir       |         path to the folder where trained model will be stored. |


##### Optional arguments

| argument | description |
| --- | --- |
| -h, --help | show help message and exit |
| -d SEED, --seed SEED | random seed passed to `pytorch`. |
| -V VALIDATION_SET, --validation-set VALIDATION_SET | comma-separated list of sample IDs to be included in the validation set. If not provided, a random validation set will be created. See `--validation-set-fraction` to choose the fraction of samples used as validation. |
| -f VALIDATION_SET_FRACTION, --validation-set-fraction VALIDATION_SET_FRACTION | fraction of samples within each species to be held as a random validation set. (default: 0.2) |
| -m PRETRAINED_MODEL, --pretrained-model PRETRAINED_MODEL | pickle file with optional pretrained neural network model to update with new images. By default models are initialized with random weigths. This option can be useful to update models as more samples are obtained. |
| -b MAX_BATCH_SIZE, --max-batch-size MAX_BATCH_SIZE | maximum batch size when using GPU for training. (default: 64) |
| -r BASE_LEARNING_RATE, --base_learning_rate BASE_LEARNING_RATE | base learning rate used in training. See https://walkwithfastai.com/lr_finder for information on learning rates. (default: 0.001) |
| -e EPOCHS, --epochs EPOCHS | number of epochs to train. See https://docs.fast.ai/ca llback.schedule.html#learner.fine_tune (default: 20) |
| -z FREEZE_EPOCHS, --freeze-epochs FREEZE_EPOCHS | number of freeze epochs to train. Recommended if using a pretrained model, but probably unnecessary if training from scratch. See https://docs.fast.ai/callback. schedule.html#learner.fine_tune (default: 0) |
| -r ARCHITECTURE, --architecture ARCHITECTURE | model architecture. See https://github.com/rwightman/pytorch-image-models for possible options. (default: ig_resnext101_32x8d) |
| -X MIX_AUGMENTATION, --mix-augmentation MIX_AUGMENTATION | apply MixUp or CutMix augmentation. Possible values are `CurMix`, `MixUp` or `None`. See https://docs.fast.ai/callback.mixup.html (default: CutMix) |
| -s, --label-smoothing | turn on Label Smoothing. See https://github.com/fastai /fastbook/blob/master/07_sizing_and_tta.ipynb (default: False) |
| -p P_LIGHTING, --p-lighting P_LIGHTING | probability of a lighting transform. Set to 0 for no lighting transforms. See https://docs.fast.ai/vision.a ugment.html#aug_transforms (default: 0.75) |
| -l MAX_LIGHTING, --max-lighting MAX_LIGHTING | maximum scale of lighting transform. See https://docs. fast.ai/vision.augment.html#aug_transforms (default: 0.5) |
| -g, --no-logging  | hide fastai progress bar and logging during training. These are shown by default. | 

#### Train command tips

All optional arguments are set to defaults that seemed to work well in our tests. Here we give some tips that may help you to modify these defaults

 1. The maximum batch size can be increased until the limit that your GPU memory supports. Larger batch sizes increase training speed, but might need adjustment of the base learning rate (which typically have to increase for large batch sizes as well). When there are only a few images available in the training set, `varKoder` automatically decreases batch size so that each training epoch sees about 10 batches of images. In our preliminary tests, we found that this improved training of these datasets.
 2. The number of epochs to train is somewhat of an art. Models trained for too long may overfit: be very good at the specific training set but bad a generalizing to new samples. Check the evolution of training and validation loss during training: if the training loss decreases but validation loss starts to increase, this means that your model is overfitting and you are training for too long. Because we introduce some random noise during training with CutMix, lighting transformations and sample subsampling, models should rarely overfit.
 3. Frozen epochs are epochs of training in which the deeper layers of a model are frozen (i. e. cannot be updated). Only the last layer is updated. This can be useful if you have a model pretrained with `varKodes` and want to use transfer learning (i. e. update a trained model instead of start from scratch). We did not find transfer learning useful when models where previously trained with other kinds of images.
 4. Finding a good learning rate is also somewhat of an art: if learning rates are too small, a model can get stuck in local optima or take too many epochs to train, wasting resources. If they are too large, the training cycle may never be able to hone into the best model weights. Our default learning rate (1e-3) behaves well for the varKodes that we used as test, but you may consider changing it in the following cases:
   1. If using a pretrained model, you may want to decrease the learning rate, since you expect to be closer to the optimal model already.
   2. If using a buch larger batch size, you may want to increase the learning rate.
 5. There is a wide array of possible model architectures, and new models come up all the time. You can use this resource to explore potential models: https://rwightman.github.io/pytorch-image-models/results/. The model we chose (ig_resnext101_32x8d) was the most accurate among those that we tested. Generally, larger models will be more accurate but need more compute resources (GPU memory and processing time).
 6. In the paper, we found that a combination of CutMix with random lighting transforms (brightness and contrast) improves training and yields more accurate models. Both are turned on by default, but you can turn them off or even change the probability that a lighting transform is applied to a varKode during training. We also tested Label Smoothing and MixUp, and these were not as helpful. For this reason, they are turned off by default but can be turned on if desired.

During training, fastai outputs a log with some information (unless you use the `-g` option). This is a table showing, for each training epoch, the loss in images in the training set (`train_loss`), the loss in images in the validation set (`valid_loss`), the accuracy in images in the validation set (`accuracy`) and the time taken for each epoch. In each epoch, the model is presented and updated with all images in the training set, split in a number of batches according to the chosen batch size. The loss is a number calculated with a loss function, which basically shows how well a neural network can predict the labels of images it is presented with. It is expected that the train loss will be small, since these are the vey same images that are used in training, and what you want is to see a small validation loss and large validation accuracy, since this shows how well your model can generalize to unseen data.

#### Output

At the end of the training cycle, three files will be written to the utput folder selected by the user:
 - `trained_model.pkl`: the model weigths exported using `fastai`. This can be used as input again using the `--pretrained-model` option in cae you want to further train the model or improve it with new images.
 - `labels.txt`: a text file with the taxon labels that can be predicted using this model.
 - `input_data.csv`: a table with information about varKodes used in the training and validation sets.

### varKoder.py query

Once of have a trained neural network, you can use the `query` command to use it to predict the taxon of unknown samples. 

#### Input format

The input for the query command is the path to a folder containing fastq or image files. There are 3 possiblities to structure this input, detailed below:

##### Single read files
If the input folder contains raw reads in fastq format (either gzipped or not), each fastq file will be considered as an independent query to build varKodes and predict their taxonomy.

##### Paired read files
If the input folder contains subfolders and each subfolder contains one or more fastq files (gzipped or not), each subfolder will be considered an independent query and the varKode will be built from all fastq files contained in each subfolder. Paired reads may be merged, simiarly to the `image` command. One model prediction will be made for each varKode (i. e. each subfolder)

##### varKodes
If the input folder contains images in the `png` format, we will assume these are varKodes and use them directly in model prediction.

#### Arguments

##### Required arguments
| argument | description |
| --- | --- |
|  model  |                pickle file with exported trained model. |
|  input  |                path to folder with fastq files to be queried. |
|  outdir  |               path to the folder where results will be saved. | 
##### Optional arguments
| argument | description |
| --- | --- |
| `-h`, `--help` | show help message and exit. |
| `-d SEED`, `--seed SEED` |  optional random seed to make sample preprocessing reproducible. |
| `-v`, `--verbose` |  show output for `fastp`, `dsand `bbtools`. By default these are ommited. This may be useful in debugging if you get errors. |
| `-k KMER_SIZE`, `--kmer-size KMER_SIZE` | size of kmers to count. Sizes from 5 to 9 are supported at the moment. (default: 7) |
| `-n N_THREADS`, `--n-threads N_THREADS` | number of samples to preprocess in parallel. See tips in `image` command on usage. (default: 1) |
| `-c CPUS_PER_THREAD`, `--cpus-per-thread CPUS_PER_THREAD` | number of cpus to use for preprocessing each sample. See tips in `image` command on usage (default: 1) |
| `-f STATS_FILE`, `--stats-file STATS_FILE`} | path to file where sample statistics will be saved. See *Output* below for details (default: stats.csv) |
| `-i INT_FOLDER`, `--int-folder INT_FOLDER` | folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used. See *Output* below for details. |
| `-m`, `--keep-images` |       save varKode image files. By default only predictions are saved and images are discarded. |
| `-a`, `--no-adapter` |      do not attempt to remove adapters from raw reads. See tips in `image` command  for details. |
| `-r`, `--no-merge` |        do not attempt to merge paired reads. See tips in `image` command  for details.|
| `-M MAX_BP`, `--max-bp MAX_BP` | maximum number of post-cleaning basepairs to make an image. By default this is all the data available for each sample. You can use SI abbreviations (e. g. 1M for 1 million or 150K for 150 thousand bp) |
| -b MAX_BATCH_SIZE, --max-batch-size MAX_BATCH_SIZE | maximum batch size when using GPU for prediction. (default: 64) |

#### Query command tips

The query command preprocesses samples to generate varKode images and then predicts their taxonomy by using a pretrained neural network. See `image` command tips for options  `--no-merge`, `--no-adapter`, `--stats-file`, `--int-folder` , `--cpus-per-thread` and `--kmer-size`.

If `--max-bp` is less than the data available for a sample, *varKoder* will ramdomly choose reads to include. If it is more than the data available for a sample, this sample will be skipped.

If there are less than 100 samples included in a query, we use a CPU to compute predictions. If there are more than 100 samples and a GPU is available, we use a GPU and group varKodes in batches of size `--max-batch-size`. The only constraint to batch size is the memory available in the GPU: the larger the batch size, the faster predictions will be done.

#### Output

The main output is a table in `csv` format saved as `predictions.csv` in the output folder. This includes the following columns:

 * `sample_id`: An identifier for each sample, inferred from the input file paths.
 *  `varKode_image_path`: path to varKodes used in prediction (they are deleted if inputs as raw reads and the option `--keep-images` is not used).
 *  `basepairs_used`: amount of data used to produce varKodes for query.
 *  `best_pred_label`: the best taxonomic prediction.
 *  `best_pred_prob`: the probability of the best prediction.
 *  other columns: probabilities of each taxonomic prediction.

In addition to this output table, varKodes produced from a raw reads input can be saved to the same folder with the option `--keep-images` and intermediate files will be stored in the folder provided with `--int-folder` if this option is used. Naming conventions for varKode image files are described in the `image` command above. 

## Example workflow

Include here an example of the full workflow, including image, train and query. Maybe we could include a small dataset to run the example?

## Author

B. de Medeiros ...

## Contributors

...

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


