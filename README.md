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

This command processes raw sequencing reads and produces *varKodes*, which are images representing the variation in K-mer frequencies for a given k-mer length.

Processing includes the following steps:

 - Raw read cleaning: adapter removal, overlapping pair merging, exact duplicate removal, poly-G tail trimming.
 - Raw read subsampling: random subsampling of raw read files into files with fewer number of reads. This is useful in machine learning training as a data augmentation technique to make sure inferences are robust to random variations in sequencing and amount of input data.
 - Kmer counting: count of kmers from subsampled files.
 - VarKode generation: generation of images from kmer counts, our *varKodes* that represent the genome composition of a taxon.
 - Adding metadata: labels provided by the user are saved as metadata within each image file. We also save a flag about the DNA quality.

Optionally, the command can be used to preprocess sequences without creating images. 

#### Input format

`varKoder.py image` accepts two input formats: folder-based or table-based.

Only the table-based format supports multiple labels per sample (for example, you could label the family, genus and species for a sample at the same time). With folder-based programs, varKoder will infer the label from the folder structure.

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
 
- The first folder level indicates taxon name, or any label that will be used to train the neural newtork. It can contain special characters but not `@`, `~` or `+`.
- The second folder level groups all `fastq` files for a given sample. Sample names can contain special characters but not `@`, `~` or `+`.
- Both compressed and uncompressed `fastq` files are accepted. Compressed files must end with the extension `.gz`. 
- Read pairs must have corresponding root file names and contain an indication of whether they are read 1 or 2. All examples above work for that end.
- Read files without `.1.`, `_1_`, `R1`, etc. will be considered unpaired.

##### table input format

In this case, you must provide a table in `csv` format linking a sample to its `fastq` files to sample metadata.
The required column names are `labels`,`sample` and `files`. Just like with folder input format, sample names and labels cannot use characters `@`, `~` or `+`. 

If you want to provide multiple labels or multiple files for a sample, you can use multiple lines per sample or a single line with `;` as a separator. For example, these 3 csv files would provide the same data to varKoder:

**multiple files and labels listed in a single row**
```csv
labels,sample,files
genus:Acridocarpus;species:ugandensis,ugandensis_12,12_GGGAGCTAGTGG.R1.fastq.gz;12_GGGAGCTAGTGG.R2.fastq.gz
genus:Acridocarpus;species:smeathmanii,smeathmannii_168,168_TATGTCACATGG.R1.fastq.gz;68_TATGTCACATGG.R2.fastq.gz
genus:Acridocarpus;species:macrocalyx,macrocalyx_176,176_GGACATGACCGG.R1.fastq.gz;176_GGACATGACCGG.R2.fastq.gz
```
**multiple labels listed in a single line, one file per row**
```csv
labels,sample,files
genus:Acridocarpus;species:ugandensis,ugandensis_12,12_GGGAGCTAGTGG.R1.fastq.gz
genus:Acridocarpus;species:smeathmanii,smeathmannii_168,68_TATGTCACATGG.R2.fastq.gz
genus:Acridocarpus;species:macrocalyx,macrocalyx_176,176_GGACATGACCGG.R1.fastq.gz
genus:Acridocarpus;species:ugandensis,ugandensis_12,12_GGGAGCTAGTGG.R2.fastq.gz
genus:Acridocarpus;species:smeathmanii,smeathmannii_168,168_TATGTCACATGG.R1.fastq.gz
genus:Acridocarpus;species:macrocalyx,macrocalyx_176,176_GGACATGACCGG.R2.fastq.gz
```

**multiple labels and files per sample, listed in separate rows**
```csv
labels,sample,files
genus:Acridocarpus,ugandensis_12,12_GGGAGCTAGTGG.R1.fastq.gz
species:smeathmanii,smeathmannii_168,68_TATGTCACATGG.R2.fastq.gz
genus:Acridocarpus,macrocalyx_176,176_GGACATGACCGG.R1.fastq.gz
species:ugandensis,ugandensis_12,12_GGGAGCTAGTGG.R2.fastq.gz
genus:Acridocarpus,smeathmannii_168,168_TATGTCACATGG.R1.fastq.gz
species:macrocalyx,macrocalyx_176,176_GGACATGACCGG.R2.fastq.gz
```

In this case, the model will try to predict both the genus and the species for each sample. You do not need to provide multiple labels per sample, or explicitly include taxonomic level. If you only wants to predict genera, for example, this would work:

```csv
labels,sample,files
Acridocarpus,ugandensis_12,12_GGGAGCTAGTGG.R1.fastq.gz;12_GGGAGCTAGTGG.R2.fastq.gz
Acridocarpus,smeathmannii_168,168_TATGTCACATGG.R1.fastq.gz;68_TATGTCACATGG.R2.fastq.gz
Acridocarpus,macrocalyx_176,176_GGACATGACCGG.R1.fastq.gz;176_GGACATGACCGG.R2.fastq.gz
```

Note:
 - file paths must be given relative to the folder where the `csv` file is located (in the example above, they are all in the same folder).
 
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
| `-k KMER_SIZE`, `--kmer-size KMER_SIZE` | length of kmers to count. Lengths from 5 to 9 are supported at the moment. (default: 7) |
| `-n N_THREADS`, `--n-threads N_THREADS` | number of samples to preprocess in parallel. See tips below on usage. (default: 1) |
| `-c CPUS_PER_THREAD`, `--cpus-per-thread CPUS_PER_THREAD` | number of cpus to use for preprocessing each sample. See tips below on usage (default: 1) |
| `-o OUTDIR`, `--outdir OUTDIR` | path to folder where to write final images. (default: images) |
| `-f STATS_FILE`, `--stats-file STATS_FILE`} | path to file where sample statistics will be saved. See *Output* below for details (default: stats.csv) |
| `-i INT_FOLDER`, `--int-folder INT_FOLDER` | folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used. See *Output* below for details. |
| `-m MIN_BP`, `--min-bp MIN_BP` | minimum number of post-cleaning basepairs to make an image. If any sample has less data than this threshold, this will result in an error. You can use SI abbreviations (e. g. 1M for 1 million or 150K for 150 thousand bp) (default: 500K) |
| `-M MAX_BP`, `--max-bp MAX_BP` | maximum number of post-cleaning basepairs to make an image. By default this is all the data available for each sample. You can use SI abbreviations (e. g. 1M for 1 million or 150K for 150 thousand bp) |
| `-t`, `--label-table` |  output a table with labels associated with each image, in addition to including them in the EXIF data. |
| `-a`, `--no-adapter` |      do not attempt to remove adapters from raw reads. See notes below for details. |
| `-r`, `--no-merge` |        do not attempt to merge paired reads. See notes below for details. |
| `-X`, `--no-image` |       clean and split raw reads, but do not generate image. You must provide a folder to save intermediate files with `--int-folder` to keep the processed reads. |

#### Image command tips

The defaults for optional arguments were chosen based on our sets. Here are some tips to help you choose values for optional arguments in case you want to change them:

 1. `--min-bp` and `--max-bp`. These arguments define how many subsampled images will be generated for each sample. We use several varKodes per sample with different amounts of input data so that the neural network can be trained to ignore random fluctuations in kmer counts and focus on features that define taxa. The rule used by `varKoder.py` is to create images for subsampled sequence files with standardized input ampounts of data, between `--min-bp` amd `--max-bp` with bp amounts corresponding to 1, 2, or 5 in each order of magnitude. For example, if `--min-bp` is 1M (1 million) and `--max-bp` is 100M (100 million), you will get 7 varKode images for each sample, corresponding to randomly chosen reads for input amounts 1M, 2M, 5M, 10M, 20M, 50M and 100M of base pairs. If `--max-bp` is ommitted, we will follow this rule until we reach the largest input amount possible for the number of raw reads in each sample, including a varKode based on all of the data available.
 
 2. If `--max-bp` is set, `varKoder` first truncates raw read files to 5 times the value of `--max-bp`. This can speed up raw read cleaning and kmer counting, but it means that subsampled files are randomly chosen from the reads available in this trimmed raw read files, not the whole initial raw reads.

 3. `--no-merge` and `--no-adpater`. We have not extensively tested the effect of skipping adapter trimming and merging of overlapping reads. It may be the case that these preprocessing steps are unnecessary and you can speed up computation by using these arguments and skipping these steps.

 4. If `--n-threads` is more than 1, `varKoder` will use Python `multiprocessing` library to do sample preprocessing in parallel (i. e. clean, split raw reads and generate images). If `-cpus-per-thread` is more than one, the number of CPUs will be passed to subprograms (i. e. `fastp`, `dsk`, `bbtools`) when processing a sample: this is the number of cores dedicated to each sample. So a user can select to parallelize computing for each sample, or to do more than one sample in parallel, or both. We have not extensively tested the potential speed ups of each method.


#### Output

varKodes will be saved as png images to a folder named `images` (or another name is the `--outdir` argument is provided). Each image will be named with the following convention:

```sample@[thousands_of_bp]K+k[kmer_length].png```

For example:

```ugandensis_12@00010000K+k7.png``` is a varKode of the species `Acridocarpus`, sample id `ugandensis_12`, made from 10 million base pairs and for a kmer length of `7`.

The labels associated with this image will be saved by default as image EXIF metadata with the key `varkoderKeywords`. These include, for example, the taxon name **Acridocarpus** and a flag about the DNA quality. The metadata will be read by `varkoder.py` during training time and can also be accessed with programs such as [exiftool](https://exiftool.org/) or your [operating system](https://www.adobe.com/creativecloud/file-types/image/raster/exif-file.html).

If the option `--label-table` has been used, these labels will be additionally saved in the output folder in a csv file named `labels.csv`.

`varkoder.py image` will also save a csv table with sample processing statistics. By default, this is `stats.csv` but it can be changed with the `--stats-file` argument.

By default, only the stats file and final images are saved. Intermediate files (clean reads, `fastp` reports, processed reads, kmer counts) are saved in a temporary folder and deleted when `varKoder` finishes processing. To save these files instead, provide a folder path with the `--int-folder` argument. In the provided path, `varKoder` will save 3 folders: 

 `clean_reads`: fastq files adapter removed and merged (also trimmed if `--max-bp` provided)
 `split_fastqs`: subsampled fastq files from clean reads
 `Xmer_counts`: `dsk` kmer count files (with `X` being the kmer length)
 
##### Note on quality labeling:

When producing *varKodes*, we use `fastp` xml output to evaluate sequence quality. fastp outputs average base pair frequecies for each position along reads. Briefly, in high-quality sample, we expect that base pair frequencies do not change throughout a read. This is because reads are randomly placed throughout a genome, so the base pair frequencies in each position will converge to the genomic composition. However, multiple process that can affect the quality of a prediction can change that. For example:

 * DNA damage causes an increase in C to T substitutions towards the end of reads (see https://doi.org/10.1098/rsos.160239)
 * Low diversity libraries might consistently start in certain genomic regions instead of randomly

Because this is fast and easy to calculate from files produced during sequencing processing, we use the variation of average base pair frequencies along reads as a heuristics to flag samples that may be low quality. The image classification model then is able to learn the features of these low-quality samples independently from their taxonomy, decreasing error rates.


### varKoder.py train

After *varKodes* are generated with `varKoder.py image`, they can be used to train a neural network to recognize taxa based on these images. The `varKoder.py train` command uses `fastai` and `pytorch` to do this training, with image models obtained with the `timm` library. If a model in the `timm` library requires a specific input image size (for example, [vision transformers](https://huggingface.co/google/vit-base-patch16-224-in21k)), **varKoder** will automatically resize input **varkodes** using the [nearest pixel method](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.Resampling.NEAREST).

There are two modes of training:

 1. Multi-label
 
 This is the default training mode. With [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification), each *varKode* can be associated with one or more labels. For example, you can use this to provide more information about the sample in addition to its identification, or to provide a full taxonomic hierarchy instead of a single name. In our tests, we have found that we can increase accuracy of predictions by flagging samples likely to have low DNA quality.
 
 2. Single label
 
  Our initial tests were all done with single-label classification. Even though we found limitation with this mode, we keep the option to do it for compatibility. In this case, it is not possible to account for sample quality when making predictions. To enable single label classification, you have to use options `--single-label` and `--ignore-quality`. 


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
| -t LABEL_TABLE, --label-table LABEL_TABLE | path to csv table with labels for each sample. If not provided, varKoder will attemp to read labels from the image file metadata. |
| -i, --ignore-quality |  ignore sequence quality when training. By default low-quality samples are labelled as such. |
|  -n, --single-label  |  Train as a single-label image classification model. This option must be combined with --ignore-quality. By default, models are trained as multi-label. |
|  -d THRESHOLD, --threshold THRESHOLD | Confidence threshold to calculate validation set metrics during training. Ignored if using --single-label (default: 0.7) |
| -V VALIDATION_SET, --validation-set VALIDATION_SET | comma-separated list of sample IDs to be included in the validation set. If not provided, a random validation set will be created. See `--validation-set-fraction` to choose the fraction of samples used as validation. |
| -f VALIDATION_SET_FRACTION, --validation-set-fraction VALIDATION_SET_FRACTION | fraction of samples to be held as a random validation set. If using multi-label, this applies to all samples. If using single-label, this applies to each species. (default: 0.2) |
| -m PRETRAINED_MODEL, --pretrained-model PRETRAINED_MODEL | pickle file with optional pretrained neural network model to update with new images. By default models are initialized with random weights. This option can be useful to update models as more samples are obtained. |
| -b MAX_BATCH_SIZE, --max-batch-size MAX_BATCH_SIZE | maximum batch size when using GPU for training. (default: 64) |
| -r BASE_LEARNING_RATE, --base_learning_rate BASE_LEARNING_RATE | base learning rate used in training. See https://walkwithfastai.com/lr_finder for information on learning rates. (default: 0.005) |
| -e EPOCHS, --epochs EPOCHS | number of epochs to train. See https://docs.fast.ai/ca llback.schedule.html#learner.fine_tune (default: 20) |
| -z FREEZE_EPOCHS, --freeze-epochs FREEZE_EPOCHS | number of freeze epochs to train. Recommended if using a pretrained model, but probably unnecessary if training from scratch. See https://docs.fast.ai/callback. schedule.html#learner.fine_tune (default: 0) |
| -c ARCHITECTURE, --architecture ARCHITECTURE | model architecture. See https://github.com/rwightman/pytorch-image-models for possible options. (default: ig_resnext101_32x8d) |
| -P, --pretrained | download pretrained model weights from timm. See https://github.com/rwightman/pytorch-image-models. (default: False) |
| -X MIX_AUGMENTATION, --mix-augmentation MIX_AUGMENTATION | apply MixUp or CutMix augmentation. Possible values are `CurMix`, `MixUp` or `None`. See https://docs.fast.ai/callback.mixup.html (default: MixUp) |
| -s, --label-smoothing | turn on Label Smoothing. Only applies to single-label. See https://github.com/fastai /fastbook/blob/master/07_sizing_and_tta.ipynb (default: False) |
| -p P_LIGHTING, --p-lighting P_LIGHTING | probability of a lighting transform. Set to 0 for no lighting transforms. See https://docs.fast.ai/vision.a ugment.html#aug_transforms (default: 0.75) |
| -l MAX_LIGHTING, --max-lighting MAX_LIGHTING | maximum scale of lighting transform. See https://docs. fast.ai/vision.augment.html#aug_transforms (default: 0.25) |
| -g, --no-logging  | hide fastai progress bar and logging during training. These are shown by default. | 

#### Train command tips

All optional arguments are set to defaults that seemed to work well in our tests. Here we give some tips that may help you to modify these defaults

 1. The maximum batch size can be increased until the limit that your GPU memory supports. Larger batch sizes increase training speed, but might need adjustment of the base learning rate (which typically have to increase for large batch sizes as well). When there are only a few images available in the training set, `varKoder` automatically decreases batch size so that each training epoch sees about 10 batches of images. In our preliminary tests, we found that this improved training of these datasets.
 2. The number of epochs to train is somewhat of an art. Models trained for too long may overfit: be very good at the specific training set but bad a generalizing to new samples. Check the evolution of training and validation loss during training: if the training loss decreases but validation loss starts to increase, this means that your model is overfitting and you are training for too long. Because we introduce random noise during training with MixUp and lighting transformations, models should rarely overfit.
 3. Frozen epochs are epochs of training in which the deeper layers of a model are frozen (i. e. cannot be updated). Only the last layer is updated. This can be useful if you have a model pretrained with `varKodes` and want to use transfer learning (i. e. update a trained model instead of start from scratch). We did not find transfer learning useful when models where previously trained with other kinds of images.
 4. Finding a good learning rate is also somewhat of an art: if learning rates are too small, a model can get stuck in local optima or take too many epochs to train, wasting resources. If they are too large, the training cycle may never be able to hone into the best model weights. Our default learning rate (5e-3) behaves well for the varKodes that we used as test, but you may consider changing it in the following cases:
   1. If using a pretrained model, you may want to decrease the learning rate, since you expect to be closer to the optimal model already.
   2. If using a much larger batch size, you may want to increase the learning rate.
 5. There is a wide array of possible model architectures, and new models come up all the time. You can use this resource to explore potential models: https://rwightman.github.io/pytorch-image-models/results/. The model we chose (ig_resnext101_32x8d) was the most accurate among those that we tested. Generally, larger models will be more accurate but need more compute resources (GPU memory and processing time).
 6. In the paper, we found that a combination of CutMix with random lighting transforms (brightness and contrast) improves training and yields more accurate models for single-label models. MixUp had a similar performance to CutMix, and it seemed to work much better for multi-label classification. For this reason, MixUp and lighting transforms are turned on by default, but you can turn them off or even change the probability that a lighting transform is applied to a *varKode* during training. We also tested Label Smoothing, which was not as helpful. For this reason, it is turned off by default but can be turned on if desired.

During training, fastai outputs a log with some information (unless you use the `-g` option). This is a table showing, for each training epoch, the loss in images in the training set (`train_loss`), the loss in images in the validation set (`valid_loss`), the accuracy in images in the validation set. In the case of multi-label models, accuracy is measured as [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) using the provided confidence threshold for predictions and ignoring DNA quality labels. In the case of single-label models, we report `accuracy`, which is the fraction of varKodes for which the correct label is predicted. In each epoch, the model is presented and updated with all images in the training set, split in a number of batches according to the chosen batch size. The loss is a number calculated with a loss function, which basically shows how well a neural network can predict the labels of images it is presented with. It is expected that the train loss will be small, since these are the very same images that are used in training, and what you want is to see a small validation loss and large validation accuracy, since this shows how well your model can generalize to unseen data.

#### Output

At the end of the training cycle, three files will be written to the utput folder selected by the user:
 - `trained_model.pkl`: the model weigths exported using `fastai`. This can be used as input again using the `--pretrained-model` option in cae you want to further train the model or improve it with new images.
 - `labels.txt`: a text file with the labels that can be predicted using this model.
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
| `-I`, `--images` |  input folder contains processed images instead of raw reads. (default: False). If you use this flag, all options for sequence processing will be ignored and `varKoder` will look for png files in the input folder. It will report the predictions for these png files. |
| `-k KMER_SIZE`, `--kmer-size KMER_SIZE` | size of kmers to count. Sizes from 5 to 9 are supported at the moment. (default: 7) |
| `-n N_THREADS`, `--n-threads N_THREADS` | number of samples to preprocess in parallel. See tips in `image` command on usage. (default: 1) |
| `-c CPUS_PER_THREAD`, `--cpus-per-thread CPUS_PER_THREAD` | number of cpus to use for preprocessing each sample. See tips in `image` command on usage (default: 1) |
| `-f STATS_FILE`, `--stats-file STATS_FILE`} | path to file where sample statistics will be saved. See *Output* below for details (default: stats.csv) |
| `-t THRESHOLD`, `--threshold THRESHOLD`} | Threshold to make a prediction. This is the minimum confidence necessary (one a scale 0-1) for varKoder to predict a taxon or other label for a given sample. (default: 0.5) |
| `-i INT_FOLDER`, `--int-folder INT_FOLDER` | folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used. See *Output* below for details. |
| `-m`, `--keep-images` |       save varKode image files. By default only predictions are saved and images are discarded. |
| `-a`, `--no-adapter` |      do not attempt to remove adapters from raw reads. See tips in `image` command  for details. |
| `-r`, `--no-merge` |        do not attempt to merge paired reads. See tips in `image` command  for details.|
| `-M MAX_BP`, `--max-bp MAX_BP` | maximum number of post-cleaning basepairs to make an image. By default this is all the data available for each sample. You can use SI abbreviations (e. g. 1M for 1 million or 150K for 150 thousand bp) |
| `-b MAX_BATCH_SIZE`, `--max-batch-size MAX_BATCH_SIZE` | maximum batch size when using GPU for prediction. (default: 64) |

#### Query command tips

The query command preprocesses samples to generate varKode images and then predicts their taxonomy by using a pretrained neural network. See `image` command tips for options  `--no-merge`, `--no-adapter`, `--stats-file`, `--int-folder` , `--cpus-per-thread` and `--kmer-size`.

If `--max-bp` is less than the data available for a sample, *varKoder* will ramdomly choose reads to include. If it is more than the data available for a sample, this sample will be skipped.

If there are less than 100 samples included in a query, we use a CPU to compute predictions. If there are more than 100 samples and a GPU is available, we use a GPU and group varKodes in batches of size `--max-batch-size`. The only constraint to batch size is the memory available in the GPU: the larger the batch size, the faster predictions will be done.

#### Output

The main output is a table in `csv` format saved as `predictions.csv` in the output folder. The columns included depend on whether the model used for predictions is single-label or multi-label. In addition to this output table, varKodes produced from a raw reads input can be saved to the same folder with the option `--keep-images` and intermediate files will be stored in the folder provided with `--int-folder` if this option is used. Naming conventions for varKode image files are described in the `image` command above. 

##### Multi-label:

 
 *  `varKode_image_path`: path to varKodes used in prediction (they are deleted if inputs as raw reads and the option `--keep-images` is not used).
 *  `sample_id`: An identifier for each sample, inferred from the input file paths.
 *  `query_basepairs`: amount of data used to produce varKodes for query.
 *  `query_kmer_len`: kmer length used to produce varKode.
 *  `trained_model_path`: path to model used to make predictions.
 *  `prediction_type`: Multilabel
 *  `prediction_threshold`: Confidence threshold to call a label
 *  `predicted_labels`: labels above the confidence threshold.
 *  `actual_labels`: labels in the EXIF metadata of a given varKode file. These are not used in the query command, just reported for comparison.
 *  other columns: confidence scores in each label. Each confidence score varies independently between 0 and 1.


##### Single-label:

 *  `varKode_image_path`: path to varKodes used in prediction (they are deleted if inputs as raw reads and the option `--keep-images` is not used).
 *  `sample_id`: An identifier for each sample, inferred from the input file paths.
 *  `query_basepairs`: amount of data used to produce varKodes for query.
 *  `query_kmer_len`: kmer length used to produce varKode.
 *  `trained_model_path`: path to model used to make predictions.
 *  `prediction_type`: Multilabel
 *  `best_pred_label`: the best taxonomic prediction.
 *  `best_pred_prob`: the confidence of the best prediction.
 *  `actual_labels`: labels in the EXIF metadata of a given varKode file. These are not used in the query command, just reported for comparison.
 *  other columns: confidence scores in each label. All confidence scores sum to 1


## Example workflow

Include here an example of the full workflow, including image, train and query. Maybe we could include a small dataset to run the example?

## Author

B. de Medeiros (Field Museum of Natural History), starting in 2019.


