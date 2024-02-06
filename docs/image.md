# varKoder image

This command processes raw sequencing reads and produces *varKodes*, which are images representing the variation in K-mer frequencies for a given k-mer length.

Processing includes the following steps:

 - Raw read cleaning: adapter removal, overlapping pair merging, exact duplicate removal, poly-G tail trimming.
 - Raw read subsampling: random subsampling of raw read files into files with fewer number of reads. This is useful in machine learning training as a data augmentation technique to make sure inferences are robust to random variations in sequencing and amount of input data.
 - Kmer counting: count of kmers from subsampled files.
 - VarKode generation: generation of images from kmer counts, our *varKodes* that represent the genome composition of a taxon.
 - Adding metadata: labels provided by the user are saved as metadata within each image file. We also save a flag about the DNA quality.

Optionally, the command can be used to preprocess sequences without creating images. 

## Input format

`varKoder image` accepts two input formats: folder-based or table-based.

Only the table-based format supports multiple labels per sample (for example, you could label the family, genus and species for a sample at the same time). With folder-based programs, varKoder will infer the label from the folder structure.

### folder input format

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

### table input format

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
species:ugandensis,ugandensis_12,12_GGGAGCTAGTGG.R2.fastq.gz
genus:Acridocarpus,smeathmannii_168,168_TATGTCACATGG.R1.fastq.gz
species:smeathmanii,smeathmannii_168,68_TATGTCACATGG.R2.fastq.gz
genus:Acridocarpus,macrocalyx_176,176_GGACATGACCGG.R1.fastq.gz
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
 
## Arguments

### Required arguments

| argument | description |
| --- | --- |
| `input` | path to either the folder with fastq files or csv file relating file paths to samples. See input formats above. |

### Optional arguments

| argument | description |
| --- | --- |
| `-h`, `--help` | show help message and exit. |
| `-d SEED`, `--seed SEED` |  optional random seed to make sample preprocessing reproducible. |
| `-v`, `--verbose` |  show output for `fastp`, `dsk` and `bbtools`. By default these are ommited. This may be useful in debugging if you get errors. |
| `-vv`, `--version` |  shows varKoder version. |
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
| `-D`, `--no-deduplicate` |        do not attempt to remove duplicates in reads. See notes below for details. |
| `-X`, `--no-image` |       clean and split raw reads, but do not generate image. You must provide a folder to save intermediate files with `--int-folder` to keep the processed reads. |
| `-T FRONT_BP,TAIL_BP`, `--trim-bp FRONT_BP,TAIL_BP` | number of base pairs to trim from the beginning and end of each read, separated by comma. (default: 10,10) |

## Image command tips

The defaults for optional arguments were chosen based on our sets. Here are some tips to help you choose values for optional arguments in case you want to change them:

 1. `--min-bp` and `--max-bp`. These arguments define how many subsampled images will be generated for each sample. We use several varKodes per sample with different amounts of input data so that the neural network can be trained to ignore random fluctuations in kmer counts and focus on features that define taxa. The rule used by `varKoder` is to create images for subsampled sequence files with standardized input ampounts of data, between `--min-bp` amd `--max-bp` with bp amounts corresponding to 1, 2, or 5 in each order of magnitude. For example, if `--min-bp` is 1M (1 million) and `--max-bp` is 100M (100 million), you will get 7 varKode images for each sample, corresponding to randomly chosen reads for input amounts 1M, 2M, 5M, 10M, 20M, 50M and 100M of base pairs. If `--max-bp` is ommitted, we will follow this rule until we reach the largest input amount possible for the number of raw reads in each sample, including a varKode based on all of the data available.
 
 2. If `--max-bp` is set, `varKoder` first truncates raw read files to 5 times the value of `--max-bp`. This can speed up raw read cleaning and kmer counting, but it means that subsampled files are randomly chosen from the reads available in this trimmed raw read files, not the whole initial raw reads.

 3. `--no-deduplicate`, `--no-merge` and `--no-adpater`. We have not extensively tested the effect of skipping adapter trimming, deduplication and merging of overlapping reads. It may be the case that these preprocessing steps are unnecessary and you can speed up computation by using these arguments and skipping these steps.

 4. If `--n-threads` is more than 1, `varKoder` will use Python `multiprocessing` library to do sample preprocessing in parallel (i. e. clean, split raw reads and generate images). If `-cpus-per-thread` is more than one, the number of CPUs will be passed to subprograms (i. e. `fastp`, `dsk`, `bbtools`) when processing a sample: this is the number of cores dedicated to each sample. So a user can select to parallelize computing for each sample, or to do more than one sample in parallel, or both. We have not extensively tested the potential speed ups of each method.


## Output

varKodes will be saved as png images to a folder named `images` (or another name is the `--outdir` argument is provided). To avoid file system problems, `varKoder` will automatically create a random subfolder structure if there are thousands of input samples, so that no folder has more than a few thousand files. Each image will be named with the following convention:

```sample@[thousands_of_bp]K+k[kmer_length].png```

For example:

```ugandensis_12@00010000K+k7.png``` is a varKode of the species `Acridocarpus`, sample id `ugandensis_12`, made from 10 million base pairs and for a kmer length of `7`.

The labels associated with this image will be saved by default as image EXIF metadata with the key `varkoderKeywords`. These include, for example, the taxon name **Acridocarpus** and a flag about the DNA quality. The metadata will be read by `varKoder` during training time and can also be accessed with programs such as [exiftool](https://exiftool.org/) or your [operating system](https://www.adobe.com/creativecloud/file-types/image/raster/exif-file.html).

If the option `--label-table` has been used, these labels will be additionally saved in the output folder in a csv file named `labels.csv`.

`varKoder image` will also save a csv table with sample processing statistics. By default, this is `stats.csv` but it can be changed with the `--stats-file` argument.

By default, only the stats file and final images are saved. Intermediate files (clean reads, `fastp` reports, processed reads, kmer counts) are saved in a temporary folder and deleted when `varKoder` finishes processing. To save these files instead, provide a folder path with the `--int-folder` argument. In the provided path, `varKoder` will save 3 folders: 

 `clean_reads`: fastq files adapter removed and merged (also trimmed if `--max-bp` provided)
 `split_fastqs`: subsampled fastq files from clean reads
 `Xmer_counts`: `dsk` kmer count files (with `X` being the kmer length)
 
## Note on quality labeling:

When producing *varKodes*, we use `fastp` xml output to evaluate sequence quality. fastp outputs average base pair frequecies for each position along reads. Briefly, in high-quality sample, we expect that base pair frequencies do not change throughout a read. This is because reads are randomly placed throughout a genome, so the base pair frequencies in each position will converge to the genomic composition. However, multiple process that can affect the quality of a prediction can change that. For example:

 * DNA damage causes an increase in C to T substitutions towards the end of reads (see https://doi.org/10.1098/rsos.160239)
 * Low diversity libraries might consistently start in certain genomic regions instead of randomly


Because this is fast and easy to calculate from files produced during sequencing processing, we use the variation of average base pair frequencies along reads as a heuristics to flag samples that may be low quality. Therefore, users can take that into account when evaluating predictions. 
The expected standard deviation for a high-quality, high-diversity sample is 0. We chose an arbitrary threshold of 0.01 to flag samples as low quality if they show more than that.
