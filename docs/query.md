# varKoder.py query

Once of have a trained neural network, you can use the `query` command to use it to predict the taxon of unknown samples. 

## Input format

The input for the query command is the path to a folder containing fastq or image files. There are 3 possiblities to structure this input, detailed below:

### Single read files
If the input folder contains raw reads in fastq format (either gzipped or not), each fastq file will be considered as an independent query to build varKodes and predict their taxonomy.

### Paired read files
If the input folder contains subfolders and each subfolder contains one or more fastq files (gzipped or not), each subfolder will be considered an independent query and the varKode will be built from all fastq files contained in each subfolder. Paired reads may be merged, simiarly to the `image` command. One model prediction will be made for each varKode (i. e. each subfolder)

### varKodes
If the input folder contains images in the `png` format, we will assume these are varKodes and use them directly in model prediction.

## Arguments

### Required arguments
| argument | description |
| --- | --- |
|  model  |                pickle file with exported trained model. |
|  input  |                path to folder with fastq files to be queried. |
|  outdir  |               path to the folder where results will be saved. | 
### Optional arguments
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

## Query command tips

The query command preprocesses samples to generate varKode images and then predicts their taxonomy by using a pretrained neural network. See `image` command tips for options  `--no-merge`, `--no-adapter`, `--stats-file`, `--int-folder` , `--cpus-per-thread` and `--kmer-size`.

If `--max-bp` is less than the data available for a sample, *varKoder* will ramdomly choose reads to include. If it is more than the data available for a sample, this sample will be skipped.

If there are less than 100 samples included in a query, we use a CPU to compute predictions. If there are more than 100 samples and a GPU is available, we use a GPU and group varKodes in batches of size `--max-batch-size`. The only constraint to batch size is the memory available in the GPU: the larger the batch size, the faster predictions will be done.

## Output

The main output is a table in `csv` format saved as `predictions.csv` in the output folder. The columns included depend on whether the model used for predictions is single-label or multi-label. In addition to this output table, varKodes produced from a raw reads input can be saved to the same folder with the option `--keep-images` and intermediate files will be stored in the folder provided with `--int-folder` if this option is used. Naming conventions for varKode image files are described in the `image` command above. 

### Multi-label:

 
 *  `varKode_image_path`: path to varKodes used in prediction (use `--keep-images` if the input is raw reads and you want to keep these files).
 *  `sample_id`: An identifier for each sample, inferred from the input file paths.
 *  `query_basepairs`: amount of data used to produce varKodes for query.
 *  `query_kmer_len`: kmer length used to produce varKode.
 *  `trained_model_path`: path to model used to make predictions.
 *  `prediction_type`: Multilabel
 *  `prediction_threshold`: Confidence threshold to call a label
 *  `predicted_labels`: labels above the confidence threshold.
 *  `actual_labels`: labels in the EXIF metadata of a given varKode file. These are not used in the query command, just reported for comparison.
 *  `possible_low_quality`: whether sample possibly has low quality. See [Notes on quality labelling](image.md) for details.
 *  other columns: confidence scores in each label. Each confidence score varies independently between 0 and 1.


### Single-label:

 *  `varKode_image_path`: path to varKodes used in prediction (use `--keep-images` if the input is raw reads and you want to keep these files).
 *  `sample_id`: An identifier for each sample, inferred from the input file paths.
 *  `query_basepairs`: amount of data used to produce varKodes for query.
 *  `query_kmer_len`: kmer length used to produce varKode.
 *  `trained_model_path`: path to model used to make predictions.
 *  `prediction_type`: Single label
 *  `best_pred_label`: the best taxonomic prediction.
 *  `best_pred_prob`: the confidence of the best prediction.
 *  `actual_labels`: labels in the EXIF metadata of a given varKode file. These are not used in the query command, just reported for comparison.
 *  `possible_low_quality`: whether sample possibly has low quality. See [Notes on quality labelling](image.md) for details.
 *  other columns: confidence scores in each label. All confidence scores sum to 1
