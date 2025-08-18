# varKoder query

The `query` command predicts labels based on a trained model and unknown samples. You can use `varKoder train` to train your own model, but by default a pretrained model available on huggingface hub is used. Currently, this model is [brunoasm/vit_large_patch32_224.NCBI_SRA](https://huggingface.co/brunoasm/vit_large_patch32_224.NCBI_SRA), follow the link for more details.

## Input format

The input for the query command is the path to a folder containing fastq or image files. There are 3 possibilities to structure this input, detailed below:

### Single read files
If the input folder contains raw reads in fastq format (either gzipped or not), each fastq file will be considered as an independent query to build varKodes and predict their taxonomy.

### Paired read files
If the input folder contains subfolders and each subfolder contains one or more fastq files (gzipped or not), each subfolder will be considered an independent query and the varKode will be built from all fastq files contained in each subfolder. Paired reads may be merged, similar to the `image` command. One model prediction will be made for each varKode (i. e. each subfolder)

### varKodes
If the input folder contains images in the `png` format and the option `--images` is used, we will assume these are varKodes and use them directly in model prediction. No sequence will be processed.

## Arguments

### Required arguments
| argument | description |
| --- | --- |
|  input  |                path to folder with fastq files or varKode png images to be queried. |
|  outdir  |               path to the folder where results will be saved. | 
### Optional arguments
| argument | description |
| --- | --- |
| `-h`, `--help` | show help message and exit. |
| `-d SEED`, `--seed SEED` |  optional random seed to make sample preprocessing reproducible. |
| `-x` `--overwrite` | overwrite results. | 
| `-vv`, `--version` |  shows varKoder version. |
| `-l MODEL`, `--model MODEL` | path pickle file with exported trained model or name of HuggingFace hub model (default: brunoasm/vit_large_patch32_224.NCBI_SRA) | 
| `-v`, `--verbose` |  show output for `fastp`, `dsk` and `bbtools`. By default these are ommited. This may be useful in debugging if you get errors. |
| `-1`, `--no-pairs` |  prevents varKoder query from considering folder structure in input to find read pairs. Each fastq file will be treated as a separate sample. But default, we assume that folders contain reads for each sample. | 
| `-I`, `--images` |  input folder contains processed images instead of raw reads. (default: False). If you use this flag, all options for sequence processing will be ignored and `varKoder` will look for png files in the input folder. It will report the predictions for these png files. |
| `-k KMER_SIZE`, `--kmer-size KMER_SIZE` | size of kmers to count. Sizes from 5 to 9 are supported at the moment. (default: 7) |
| `-p KMER_MAPPING`, `--kmer-mapping KMER_MAPPING` | method to map kmers to pixels if input data is sequences. This sets the correspondence between specific kmers and coordinates in the image produced. The possible values are: `varKode`, which produces varKodes; `cgr` (default), which produces a rfCGR. See the [convert command documentation](convert.md) for more details.|
| `-n N_THREADS`, `--n-threads N_THREADS` | number of samples to preprocess in parallel. See tips in `image` command on usage. (default: 1) |
| `-c CPUS_PER_THREAD`, `--cpus-per-thread CPUS_PER_THREAD` | number of cpus to use for preprocessing each sample. See tips in `image` command on usage (default: 1) |
| `-f STATS_FILE`, `--stats-file STATS_FILE`} | path to file where sample statistics will be saved. See *Output* below for details (default: stats.csv) |
| `-d THRESHOLD`, `--threshold THRESHOLD`} | Threshold to make a prediction. This is the minimum confidence necessary (one a scale 0-1) for varKoder to predict a taxon or other label for a given sample. (default: 0.7) |
| `-i INT_FOLDER`, `--int-folder INT_FOLDER` | folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used. See *Output* below for details. |
| `-m`, `--keep-images` |       save varKode image files. By default only predictions are saved and images are discarded. |
| `-P`, `--include-probs` |   whether confidence scores for each label should be included in the output. By default, only predictions above threshold are report. Be careful, this can greatly increase output file size if there are many possible labels. | 
| `-a`, `--no-adapter` |      do not attempt to remove adapters from raw reads. See tips in `image` command  for details. |
| `-r`, `--no-merge` |        do not attempt to merge paired reads. See tips in `image` command  for details.|
| `-D`, `--no-deduplicate` |        do not attempt to remove duplicates in reads. See tips in `image` command for details. |
| `-T FRONT_BP,TAIL_BP`, `--trim-bp FRONT_BP,TAIL_BP` | number of base pairs to trim from the beginning and end of each read, separated by comma. This is applied to both forward and reverse reads in the case of paired ends.  (default: 10,10) |
| `-M MAX_BP`, `--max-bp MAX_BP` | number of post-cleaning basepairs to use for making image. You can use SI abbreviations (e. g. 1M for 1 million or 150K for 150 thousand bp). Set to 0 to use all of the available data. (default: 200M) |
| `-b MAX_BATCH_SIZE`, `--max-batch-size MAX_BATCH_SIZE` | maximum batch size when using GPU for prediction. (default: 64) |

## Query command tips

The query command preprocesses samples to generate varKode images and then predicts their taxonomy by using a pretrained neural network. See [image](image.md) command tips for options  `--no-merge`, `--no-adapter`, `--stats-file`, `--int-folder` , `--no-deduplicate`, `--trim-bp`, `--cpus-per-thread` and `--kmer-size`.

If `--max-bp` is less than the data available for a sample, *varKoder* will ramdomly choose reads to include. If it is more than the data available for a sample, this sample will be skipped.

If there are less than 128 samples included in a query, we use a CPU to compute predictions. If there are 128 or more samples and a GPU is available, we use a GPU and group varKodes in batches of size `--max-batch-size`. The only constraint to batch size is the memory available in the GPU: the larger the batch size, the faster predictions will be done.

## Input 
By default, if the input folder contains subfolders, `varKoder query` will assume that raw reads in each subfolder should all be treated as a single sample (named with subfolder name). To override this behavior, use `--no-pairs`. If there are no subfolders or `--no-pairs` is used, each fastq file in the input will be treated as a separate sample (named after the file name). See help on `varKoder image` command for more information about fastq processing.

If the `--images` argument is used, `varKoder query` will not attempt to process fastq files. Instead, it will recursively search for `png` files in the input folder, assuming they are varKodes generated with `varKoder image`.

## Models
If you trained your own model with `varKoder train`, you can use this model for making predictions by providing the path with the option `--model`.

If you want to use a pytorch model from [Hugging Face hub](https://huggingface.co), you can provide the repository for this model using the same option (`--model`). The default model is a model pretrained on SRA data ([brunoasm/vit_large_patch32_224.NCBI_SRA](https://huggingface.co/brunoasm/vit_large_patch32_224.NCBI_SRA)).

## Output

The main output is a table in `csv` format saved as `predictions.csv` in the output folder. The columns included depend on whether the model used for predictions is single-label or multi-label. In addition to this output table, varKodes produced from a raw reads input can be saved to the same folder with the option `--keep-images` and intermediate files will be stored in the folder provided with `--int-folder` if this option is used. Naming conventions for varKode image files are described in the `image` command above. 

By default, only the top prediction (if single-label) or predictions above threshold (if multi-label) are included in the table. To also include the predicted confidence of all possible labels, use the argument `--include-probs`. CAUTION: if there are many possible labels in the trained model (for example, thousands) this can generate a very large output file.

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
 *  other columns: confidence scores in each label. Each confidence score varies independently between 0 and 1. They are only included with `--include-probs` option.


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
 *  other columns: confidence scores in each label. All confidence scores sum to 1. They are only included with `--include-probs` option.
 
## Examples

Here are several examples demonstrating how to use the `query` command in different scenarios:

### Example 1: Basic Query with Default Model

Query a set of fastq files using the default pretrained model:

```bash
varKoder query path/to/fastq_files query_results
```

This processes the fastq files to generate varKodes, then uses the default Hugging Face model (brunoasm/vit_large_patch32_224.NCBI_SRA) to predict labels with a confidence threshold of 0.5.

### Example 2: Using Your Custom-trained Model

Query samples using a model you trained:

```bash
varKoder query path/to/fastq_files custom_query_results --model path/to/trained_model.pkl --threshold 0.7
```

This uses your custom-trained model instead of the default one, and requires a higher confidence threshold (0.7) for making predictions.

### Example 3: Directly Querying Existing varKode Images

Query existing varKode images instead of processing fastq files:

```bash
varKoder query path/to/varkode_images direct_image_results --images --threshold 0.6
```

This skips the fastq processing step and directly uses existing varKode images for prediction with a 0.6 confidence threshold.

### Example 4: Preserving Images and Including All Probability Scores

Query samples while saving the generated varKodes and including all probability scores in the output:

```bash
varKoder query path/to/fastq_files full_results --keep-images --include-probs
```

This processes the fastq files, saves the generated varKodes to the output directory, and includes confidence scores for all possible labels in the output CSV file, not just those above the threshold.

### Example 5: Processing Large Files with Parallel Processing

Process and query large sequencing files more efficiently:

```bash
varKoder query path/to/large_fastq_files parallel_results --n-threads 4 --cpus-per-thread 2 --max-bp 5M --max-batch-size 128
```

This uses 4 parallel threads with 2 CPUs each for processing samples, limits each sample to 5 million base pairs, and uses a batch size of 128 for GPU prediction, which can significantly speed up processing of large datasets.
