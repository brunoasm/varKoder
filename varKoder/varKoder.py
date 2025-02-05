#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import functions and libraries
from varKoder.functions import *
import argparse, pkg_resources

version=pkg_resources.get_distribution("varKoder").version

def main():
    # create top-level parser with common arguments
    main_parser = argparse.ArgumentParser(
        description="varKoder: using neural networks for DNA barcoding based on variation in whole-genome kmer frequencies",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = main_parser.add_subparsers(required=True, dest="command")

    parent_parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parent_parser.add_argument("-R", "--seed", help="random seed.", type=int)
    parent_parser.add_argument(
        "-x", "--overwrite", help="overwrite existing results.", action="store_true"
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        help="verbose output to stderr to help in debugging.",
        action="store_true",
        default=False,
    )
    parent_parser.add_argument("-vv", "--version", action='version', version=f'%(prog)s {version}', help="prints varKoder version installed")
    # create parser for image command
    parser_img = subparsers.add_parser(
        "image",
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Preprocess reads and prepare images for Neural Network training.",
    )
    parser_img.add_argument(
        "input",
        help="path to either the folder with fastq files or csv file relating file paths to samples. See online manual for formats.",
    )
    parser_img.add_argument(
        "-k", "--kmer-size", help="size of kmers to count (5–9)", type=int, default=7
    )
    parser_img.add_argument(
        "-p", "--kmer-mapping", help="method to map kmers. See online documentation for an explanation.", type=str, default='cgr', choices = mapping_choices
    )
    
    parser_img.add_argument(
        "-n",
        "--n-threads",
        help="number of samples to preprocess in parallel.",
        default=1,
        type=int,
    )
    parser_img.add_argument(
        "-c",
        "--cpus-per-thread",
        help="number of cpus to use for preprocessing each sample.",
        default=1,
        type=int,
    )
    parser_img.add_argument(
        "-o",
        "--outdir",
        help="path to folder where to write final images.",
        default="images",
    )
    parser_img.add_argument(
        "-f",
        "--stats-file",
        help="path to file where sample statistics will be saved.",
        default="stats.csv",
    )
    parser_img.add_argument(
        "-i",
        "--int-folder",
        help="folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used.",
    )
    parser_img.add_argument(
        "-m",
        "--min-bp",
        type=str,
        help="minimum number of post-cleaning basepairs to make an image. Samples below this threshold will be discarded",
        default="500K",
    )
    parser_img.add_argument(
        "-M",
        "--max-bp",
        help="maximum number of post-cleaning basepairs to make an image. Use '0' to use all of the available data.",
        default="200M"
    )
    parser_img.add_argument(
        "-t",
        "--label-table",
        help="output a table with labels associated with each image, in addition to including them in the EXIF data.",
        action="store_true",
    )
    parser_img.add_argument(
        "-a",
        "--no-adapter",
        help="do not attempt to remove adapters from raw reads.",
        action="store_true",
    )
    parser_img.add_argument(
        "-D",
        "--no-deduplicate",
        help="do not attempt to remove duplicates in raw reads.",
        action="store_true",
    )
    parser_img.add_argument(
        "-r",
        "--no-merge",
        help="do not attempt to merge paired reads.",
        action="store_true",
    )
    parser_img.add_argument(
        "-X",
        "--no-image",
        help="clean and split raw reads, but do not generate image.",
        action="store_true",
    )
    parser_img.add_argument(
        "-T",
        "--trim-bp",
        help="number of base pairs to trim from the start and end of each read, separated by comma.",
        default="10,10"
    )

    # create parser for train command
    parser_train = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Train a Neural Network based on provided images.",
    )
    parser_train.add_argument("input", help="path to the folder with input images.")
    parser_train.add_argument(
        "outdir", help="path to the folder where trained model will be stored."
    )
    parser_train.add_argument(
        "-n",
        "--num-workers",
        help="number of subprocess for data loading. See https://docs.fast.ai/data.load.html#dataloader",
        default=0,
        type=int,
    )
    parser_train.add_argument(
        "-t",
        "--label-table-path",
        help="path to csv table with labels for each sample. By default, varKoder will instead read labels from the image file metadata.",
    )
    parser_train.add_argument(
        "-S",
        "--single-label",
        help="Train as a single-label image classification model instead of multilabel. If multiple labels are provided, they will be concatenated to a single label.",
        action="store_true",
    )
    parser_train.add_argument(
        "-d",
        "--threshold",
        help="Threshold to calculate precision and recall during for the validation set. Ignored if using --single-label",
        type=float,
        default=0.7,
    )
    parser_train.add_argument(
        "-V",
        "--validation-set",
        help="comma-separated list of sample IDs to be included in the validation set, or path to a text file with such a list. If not provided, a random validation set will be created. Turns off --validation-set-fraction",
    )
    parser_train.add_argument(
        "-f",
        "--validation-set-fraction",
        help="fraction of samples to be held as a random validation set. Will be ignored if --validation-set is provided.",
        type=float,
        default=0.2,
    )
    parser_train.add_argument(
        "-c",
        "--architecture",
        help="model architecture. Options include all those supported by timm library  plus 'arias2022' and 'fiannaca2018'. See documentation for more info. ",
        default="hf-hub:brunoasm/vit_large_patch32_224.NCBI_SRA",
    )
    parser_train.add_argument(
        "-m",
        "--pretrained-model",
        help="optional pickle file with pretrained model to update with new images. Turns off --architecture if used.",
    )
    parser_train.add_argument(
        "-b",
        "--max-batch-size",
        help="maximum batch size when using GPU for training.",
        type=int,
        default=64,
    )
    parser_train.add_argument(
        "-r",
        "--base-learning-rate",
        help="base learning rate used in training. See https://walkwithfastai.com/lr_finder for information on learning rates.",
        type=float,
        default=5e-3,
    )
    parser_train.add_argument(
        "-e",
        "--epochs",
        help="number of epochs to train. See https://docs.fast.ai/callback.schedule.html#learner.fine_tune",
        type=int,
        default=30,
    )
    parser_train.add_argument(
        "-z",
        "--freeze-epochs",
        help="number of freeze epochs to train. Recommended if using a pretrained model. See https://docs.fast.ai/callback.schedule.html#learner.fine_tune",
        type=int,
        default=0,
    )
    parser_train.add_argument(
        "-w",
        "--random-weights",
        help="start training with random weights. By default, pretrained model weights are downloaded from timm. See https://github.com/rwightman/pytorch-image-models.",
        action="store_true",
    )
    parser_train.add_argument(
        "-i",
        "--negative_downweighting",
        type=float,
        default=4,
        help="Parameter controlling strength of loss downweighting for negative samples. See gamma(negative) parameter in https://arxiv.org/abs/2009.14119. Ignored if used with --single-label.",
    )
    # parser_train.add_argument('-i','--downweight-quality',
    #                          help = 'use a modified loss function that downweights samples based on DNA quality. Ignored if used with --single-label.',
    #                          action = 'store_true'
    #                         )
    parser_train.add_argument(
        "-X",
        "--mix-augmentation",
        help="apply MixUp or CutMix augmentation. See https://docs.fast.ai/callback.mixup.html",
        choices=["CutMix", "MixUp", "None"],
        default="MixUp",
    )
    parser_train.add_argument(
        "-s",
        "--label-smoothing",
        help="turn on Label Smoothing. Only used with --single-label. See https://github.com/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb",
        action="store_true",
        default=False,
    )
    parser_train.add_argument(
        "-p",
        "--p-lighting",
        help="probability of a lighting transform. Set to 0 for no lighting transforms. See https://docs.fast.ai/vision.augment.html#aug_transforms",
        type=float,
        default=0.75,
    )
    parser_train.add_argument(
        "-l",
        "--max-lighting",
        help="maximum scale of lighting transform. See https://docs.fast.ai/vision.augment.html#aug_transforms",
        type=float,
        default=0.25,
    )
    parser_train.add_argument(
        "-g",
        "--no-logging",
        help="hide fastai progress bar and logging during training.",
        action="store_true",
        default=False,
    )
    parser_train.add_argument(
        "-M",
        "--no-metrics",
        help="skip calculation of validation loss and metrics during training.",
        action="store_true",
        default=False,
    )

    # create parser for query command
    parser_query = subparsers.add_parser(
        "query",
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Query a fastq file agains a trained neural network. The program will automatically deduplicate, merge overlapping reads, clean adapters, count k-mers, produce images and query this image. The response is written to stdout in json format",
    )

    parser_query.add_argument(
        "input", help="path to folder with fastq files to be queried."
    )
    parser_query.add_argument(
        "outdir", help="path to the folder where results will be saved."
    )
    parser_query.add_argument(
        "-l",
        "--model", 
        help="path pickle file with exported trained model or name of HuggingFace hub model",
        default="brunoasm/vit_large_patch32_224.NCBI_SRA"
        )
    parser_query.add_argument(
        "-1",
        "--no-pairs",
        help="prevents varKoder query from considering folder structure in input to find read pairs. Each fastq file will be treated as a separate sample",
        action="store_true",
    )
    parser_query.add_argument(
        "-I",
        "--images",
        help="input folder contains processed images instead of raw reads.",
        action="store_true",
    )
    parser_query.add_argument(
        "-k", "--kmer-size", help="size of kmers to count (5–9)", type=int, default=7
    )
    parser_query.add_argument(
        "-p", "--kmer-mapping", help="method to map kmers. See online documentation for an explanation.", type=str, default='cgr', choices = mapping_choices
    )
    
    parser_query.add_argument(
        "-n",
        "--n-threads",
        help="number of samples to preprocess in parallel.",
        default=1,
        type=int,
    )
    parser_query.add_argument(
        "-c",
        "--cpus-per-thread",
        help="number of cpus to use for preprocessing each sample.",
        default=1,
        type=int,
    )
    parser_query.add_argument(
        "-f",
        "--stats-file",
        help="path to file where sample statistics will be saved.",
        default="stats.csv",
    )
    parser_query.add_argument(
        "-d",
        "--threshold",
        help="confidence threshold to make a prediction.",
        type=float,
        default=0.7,
    )
    parser_query.add_argument(
        "-i",
        "--int-folder",
        help="folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used and deleted when done.",
    )
    parser_query.add_argument(
        "-m",
        "--keep-images",
        help="whether barcode images should be saved to a directory named 'query_images'.",
        action="store_true",
    )
    parser_query.add_argument(
        "-P",
        "--include-probs",
        help="whether probabilities for each label should be included in the output.",
        action="store_true",
    )
    parser_query.add_argument(
        "-a",
        "--no-adapter",
        help="do not attempt to remove adapters from raw reads.",
        action="store_true",
    )
    parser_query.add_argument(
        "-r",
        "--no-merge",
        help="do not attempt to merge paired reads.",
        action="store_true",
    )
    parser_query.add_argument(
        "-D",
        "--no-deduplicate",
        help="do not attempt to remove duplicates in raw reads.",
        action="store_true",
    )
    parser_query.add_argument(
        "-T",
        "--trim-bp",
        help="number of base pairs to trim from the start and end of each read, separated by comma.",
        default="10,10"
    )
    parser_query.add_argument(
        "-M",
        "--max-bp",
        help="number of post-cleaning basepairs to use for making image. Use '0' to use all of the available data.",
        default="200M"
    )
    parser_query.add_argument(
        "-b",
        "--max-batch-size",
        help="maximum batch size when using GPU for predictions.",
        type=int,
        default=64,
    )

    # create parser for convert command
    parser_cvt = subparsers.add_parser(
        "convert",
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Convert images between different kmer mappings.",
    )
    parser_cvt.add_argument(
        "-n",
        "--n-threads",
        help="number of threads to process images in parallel.",
        default=1,
        type=int,
    )
    parser_cvt.add_argument(
        "-k","--kmer-size", help="size of kmers used to produce original images. Will be inferred from file names if omitted.", type=int, default=7,choices = [5,6,7,8,9]
    )
    parser_cvt.add_argument(
        "-p","--input-mapping", help="kmer mapping of input images. Will be inferred from file names if omitted.", choices = mapping_choices
    )
    parser_cvt.add_argument(
        "-r","--sum-reverse-complements", help="When converting from CGR to varKode, add together counts from canonical kmers and their reverse complements.", action = 'store_true'
    )

    parser_cvt.add_argument(
        "output_mapping", help="kmer mapping of output images.", choices = mapping_choices
    )
    parser_cvt.add_argument(
        "input", help="path to folder with png images files to be converted."
    )
    parser_cvt.add_argument(
        "outdir", help="path to the folder where results will be saved."
    )

    

    # execution
    args = main_parser.parse_args()
    
    # parse max_bp (for compatibility with previous versions of the code)
    if args.max_bp is not None and args.max_bp == 0:
        args.max_bp = None

    # check if input directory exists
    
    if not Path(args.input).exists():
        raise Exception("Input path", args.input, "does not exist. Please check.")
    #elif not Path(args.input).is_dir():
    #    raise Exception("Input path", args.input, "is not a directory. Please check.")

    # set random seed
    try:
        set_seed(args.seed)
        np_rng = np.random.default_rng(args.seed)
    except TypeError:
        np_rng = np.random.default_rng()

    ##################################################
    # preparing images for commands 'image' or 'query'
    ##################################################

    if args.command == "image" or (args.command == "query" and not args.images):
        if args.command == "query":
            if not args.overwrite:
                if Path(args.outdir, "predictions.csv").is_file():
                    raise Exception(
                        "Output predictions file exists, use --overwrite if you want to overwrite it."
                    )
        # check if kmer size provided is supported
        if args.kmer_size not in range(5, 9 + 1):
            raise Exception("kmer size must be between 5 and 9")

        # if no directory provided for intermediate results, create a temporary one
        #  that will be deleted at the end of the program
        try:
            inter_dir = Path(args.int_folder)
        except TypeError:
            inter_dir = Path(tempfile.mkdtemp(prefix="barcoding_"))
            
        # check if output directory exists
        if not args.overwrite and Path(args.outdir).exists():
            raise Exception("Output directory exists, use --overwrite if you want to overwrite it.")
        else:
            if Path(args.outdir).is_dir():
                shutil.rmtree(Path(args.outdir))

        # set directory to save images
        if args.command == "image":
            images_d = Path(args.outdir)
        elif args.command == "query":
            if args.keep_images:
                images_d = Path(args.outdir) / "query_images"
                images_d.mkdir(parents=True, exist_ok=True)
            elif args.int_folder:
                images_d = Path(tempfile.mkdtemp(prefix="barcoding_img_"))
            else:
                images_d = inter_dir / "images"



        eprint("varKoder",version)
        eprint("Kmer size:", str(args.kmer_size))
        eprint("Processing reads and preparing images")
        eprint("Reading input data")

        ##### STEP A - parse input and create a table relating reads files to samples and taxa
        inpath = Path(args.input)
        condensed_files = process_input(
            inpath, 
            is_query=args.command == "query", 
            no_pairs=args.command == "query" and getattr(args, 'no_pairs', False)
        )
        if condensed_files.shape[0] == 0:
            raise Exception("No files found in input. Please check.")
            

        ### We will save statistics in a dictionary and then output as a table
        ### If a table already exists, read it
        ### If not, start from scratch
        all_stats = defaultdict(OrderedDict)
        stats_path = Path(args.stats_file)
        if stats_path.exists():
            all_stats.update(
                    pd.read_csv(stats_path, index_col=[0],dtype={0:str},low_memory=False).to_dict(orient="index")
            )

        ### the same kmer mapping will be used for all files, so we will use it as a global variable to decrease overhead
        kmer_mapping = get_kmer_mapping(args.kmer_size, args.kmer_mapping)

        # check if we will need multiple subfolder levels
        # this will ensure we have about 1000 samples per subfolder
        # number of actual files will depend on how many images per sample
        n_records = condensed_files.shape[0]
        subfolder_levels = math.floor(math.log(n_records/1000,16))

        # Prepare arguments for run_clean2img function
        args_for_multiprocessing = [
            (
                tup,
                kmer_mapping,
                args,
                np_rng,
                inter_dir,
                all_stats,
                stats_path,
                images_d,
                subfolder_levels
            )
            for tup in condensed_files.iterrows()
        ]

        # Single-threaded execution
        if args.n_threads == 1:
            for arg_tuple in args_for_multiprocessing:
                stats = run_clean2img(*arg_tuple)
                process_stats(
                    stats,
                    condensed_files,
                    args,
                    stats_path,
                    images_d,
                    all_stats,
                    qual_thresh,
                    labels_sep,
                )

        # Multi-threaded execution
        else:
            with multiprocessing.Pool(processes=int(args.n_threads)) as pool:
                for stats in pool.imap_unordered(
                    run_clean2img_wrapper, args_for_multiprocessing
                ):
                    process_stats(
                        stats,
                        condensed_files,
                        args,
                        stats_path,
                        images_d,
                        all_stats,
                        qual_thresh,
                        labels_sep,
                    )

        eprint("All images done, saved in", str(images_d))

    ###################
    # query command
    ###################

    if args.command == "query":
        if args.images:
            images_d = Path(args.input)
        # if we provided sequences rather than images, they were processed in the command above

        img_paths = [img for img in images_d.rglob("*.png")]

        #get metadata
        actual_labels = []
        qual_flags = []
        freq_sds = []
        sample_ids = []
        query_bp = []
        query_klen = []
        query_mapping = []
        for p in img_paths:
            try:
                labs = ";".join(get_varKoder_labels(p))
            except (AttributeError, TypeError):
                labs = np.nan

            try:
                qual_flag = get_varKoder_qual(p)
            except (AttributeError, TypeError):
                qual_flag = np.nan

            try:
                freq_sd = get_varKoder_freqsd(p)
            except (AttributeError,TypeError):
                freq_sd = np.nan

            img_metadatada = get_metadata_from_img_filename(p)

            
            sample_ids.append(img_metadatada['sample'])
            query_bp.append(img_metadatada['bp'])
            query_klen.append(img_metadatada['img_kmer_size'])
            query_mapping.append(img_metadatada['img_kmer_mapping'])
            actual_labels.append(labs)
            qual_flags.append(qual_flag)
            freq_sds.append(freq_sd)

        # Start output dataframe
        common_data = {
            "varKode_image_path": img_paths,
            "sample_id": sample_ids,
            "query_basepairs": query_bp,
            "query_kmer_len": query_klen,
            "query_mapping": query_mapping,
            "trained_model_path": str(args.model),
            "actual_labels": actual_labels,
            "possible_low_quality": qual_flags,
            "basefrequency_sd": freq_sds,
        }
            
        #Decide how to compute predictions, start learner
        n_images = len(img_paths)

        try:
            if n_images >= 128:
                eprint(n_images, "images in the input, will try to use GPU for prediction.")
                learn = load_learner(args.model, cpu=False)
            else:
                eprint(n_images, "images in the input, will use CPU for prediction.")
                learn = load_learner(args.model, cpu=True)
        except FileNotFoundError:
            eprint('Model',args.model,"not found locally, trying Hugging Face hub.")
            try: 
                learn = from_pretrained_fastai(args.model)
            except:
                raise Exception('Unable to load model',args.model,"locally or from Hugging Face Hub, please check")


        df = pd.DataFrame({"path": img_paths})
        query_dl = learn.dls.test_dl(df, bs=args.max_batch_size)

        
        #make predictions and add to output dataframe
        if "MultiLabel" in str(learn.loss_func):
            eprint(
                "This is a multilabel classification model, each input may have 0 or more predictions."
            )
            pp, _ = learn.get_preds(dl=query_dl, act=nn.Sigmoid())
            above_threshold = pp >= args.threshold
            vocab = learn.dls.vocab
            predicted_labels = [
                ";".join([vocab[idx] for idx, val in enumerate(row) if val])
                for row in above_threshold
            ]
        
            output_df = pd.DataFrame({
                **common_data,
                "prediction_type": "Multilabel",
                "prediction_threshold": args.threshold,
                "predicted_labels": predicted_labels,
            })
        
        else:
            eprint(
                "This is a single label classification model, each input may will have only one prediction."
            )
            pp, _ = learn.get_preds(dl=query_dl)
        
            best_ps, best_idx = torch.max(pp, dim=1)
            best_labels = learn.dls.vocab[best_idx]
        
            output_df = pd.DataFrame({
                **common_data,
                "prediction_type": "Single label",
                "best_pred_label": best_labels,
                "best_pred_prob": best_ps,
            })
        
        if args.include_probs:
            output_df = pd.concat([output_df, pd.DataFrame(pp, columns=learn.dls.vocab)], axis=1)

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(outdir / "predictions.csv", index=False)

    ###################
    # train command
    ###################

    if args.command == "train":
        eprint("Starting train command.")
        if not args.overwrite:
            if Path(args.outdir).exists():
                raise Exception(
                    "Output directory exists, use --overwrite if you want to overwrite it."
                )

        # 1 let's create a data table for all images.
        image_files = list()
        f_counter = 0
        for f in Path(args.input).rglob("*.png"):
            image_files.append(get_metadata_from_img_filename(f))
            f_counter += 1
            if f_counter % 1000 == 0:
                eprint(f"\rFound {f_counter} image files", end='', flush=True)
        eprint(f"\rFound {f_counter} image files", flush=True)


        if args.label_table_path:
            n_image_files = pd.DataFrame(image_files).merge(
                pd.read_csv(args.label_table_path)[
                     ["sample", "labels"]
                    #["sample", "labels", "possible_low_quality"]
                ],
                on="sample",
                how="inner",
            )
            excluded_samples = set([x["sample"] for x in image_files]) - set(n_image_files["sample"])
            eprint(len(excluded_samples),"samples excluded due to absence in provided label table.")
            if args.verbose:
                eprint('Samples excluded:\n','\n'.join(excluded_samples))
            image_files = n_image_files
        else:
            image_files = pd.DataFrame(image_files).assign(
                labels=lambda x: x["path"].apply(
                    lambda y: ";".join(get_varKoder_labels(y))
                ),
                possible_low_quality=lambda x: x["path"].apply(get_varKoder_qual),
            )

        # add quality-based sample weights
        # if args.downweight_quality:
        #    image_files = image_files.assign(
        #            sample_weights = lambda x: x['path'].apply(get_varKoder_quality_weights)
        #        )
        # else:
        #    image_files['sample_weights'] = 1

        # 2 let's add a column to mark images in the validation set according to input options
        if (
            args.validation_set
        ):  # if a specific validation set was defined, let's use it
            eprint("Spliting validation set as defined by user.")
            try:  # try to treat as a path first
                with open(args.validation_set, "r") as valsamps:
                    validation_samples = valsamps.readline().strip().split(",")
            except:  # if not, try to treat as a list
                validation_samples = args.validation_set.split(",")
        else:
            eprint(
                "Splitting validation set randomly. Fraction of samples per label combination held as validation:",
                str(args.validation_set_fraction),
            )
            validation_samples = (
                image_files[["sample", "labels"]]
                .assign(
                    labels=lambda x: x["labels"].apply(
                        lambda y: ";".join(sorted([z for z in y.split(";")]))
                    )
                )
                .drop_duplicates()
                .groupby("labels")
                .sample(frac=args.validation_set_fraction)
                .loc[:, "sample"]
            )

        image_files = image_files.assign(
            is_valid=image_files["sample"].isin(validation_samples),
            labels=lambda x: x["labels"].apply(
                lambda y: ";".join(sorted([z for z in y.split(";")]))
            ),
        )

        # 3 prepare input to training function based on options
        eprint("Setting up neural network model for training.")

        callback = {"MixUp": MixUp, "CutMix": CutMix, "None": None}[
            args.mix_augmentation
        ]

        trans = aug_transforms(
            do_flip=False,
            max_rotate=0,
            max_zoom=1,
            max_lighting=args.max_lighting,
            max_warp=0,
            p_affine=0,
            p_lighting=args.p_lighting,
        )

        # 4 if a pretrained model has been provided, load model state
        # dev = torch.device('cpu')
        model_state_dict = None

        if torch.backends.mps.is_built() or (torch.backends.cuda.is_built() 
                                             and torch.cuda.device_count()):
            print("GPU available. Will try to use GPU for processing.")
            load_on_cpu = False
        else:
            load_on_cpu = True
            print("GPU not available. Using CPU for processing.")

        if args.pretrained_model:
            eprint("Loading pretrained model from file:", str(args.pretrained_model))
            past_learn = load_learner(args.pretrained_model, cpu=load_on_cpu)
            model_state_dict = past_learn.model.state_dict()
            pretrained = False
            del past_learn

        elif not args.random_weights and not args.architecture in ('arias2022', 'fiannaca2018'):
            pretrained = True
            eprint("Starting model with pretrained weights from timm library.")
            eprint("Model architecture:", args.architecture)

        else:
            pretrained = False
            eprint("Starting model with random weights.")
            eprint("Model architecture:", args.architecture)
            
        # Check for label types and warn if there seems to be a mismatch
        if args.single_label:
            eprint("Single label model requested.")
            if (image_files["labels"].str.contains(";") == True).any():
                warnings.warn(
                    "Some samples contain more than one label. These will be concatenated. Maybe you want a multilabel model instead?",
                    stacklevel=2,
                )
        else:
            eprint("Multilabel model requested.")
            if not (image_files["labels"].str.contains(";") == True).any():
                warnings.warn(
                    "No sample contains more than one label. Maybe you want a single label model instead?",
                    stacklevel=2,
                )
        
        # Set loss function based on args.mix_augmentation
        if args.mix_augmentation == "None" and args.single_label:
            loss = CrossEntropyLoss()
        elif args.single_label:
            loss = CrossEntropyLossFlat()
        else:
            loss = AsymmetricLossMultiLabel(
              gamma_pos=0, 
              gamma_neg=args.negative_downweighting, 
              eps=1e-2, 
              clip=0.1)
        
        # Print training information
        eprint(
            "Start training for",
            args.freeze_epochs,
            "epochs with frozen model body weights followed by",
            args.epochs,
            "epochs with unfrozen weights and learning rate of",
            args.base_learning_rate,
        )
        
        # Additional parameters for multilabel training
        extra_params = {}
        if not args.single_label:
            extra_params = {
                "metrics_threshold": args.threshold,
            }
        
        # Call training function
        learn = train_nn(
            df=image_files,
            architecture=args.architecture,
            valid_pct=args.validation_set_fraction,
            max_bs=args.max_batch_size,
            base_lr=args.base_learning_rate,
            epochs=args.epochs,
            freeze_epochs=args.freeze_epochs,
            normalize=True,
            pretrained=pretrained,
            callbacks=callback,
            max_lighting=args.max_lighting,
            p_lighting=args.p_lighting,
            loss_fn=loss,
            model_state_dict=model_state_dict,
            verbose=not args.no_logging,
            is_multilabel=not args.single_label,
            num_workers=args.num_workers,
            no_metrics=args.no_metrics,
            **extra_params
        )

        # save results
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        learn.export(outdir / "trained_model.pkl")
        with open(outdir / "labels.txt", "w") as outfile:
            outfile.write("\n".join(learn.dls.vocab))
        image_files.to_csv(outdir / "input_data.csv",index=False)

        eprint("Model, labels, and data table saved to directory", str(outdir))


    
    ###################
    # convert command
    ###################
    if args.command == "convert":

        if not args.overwrite:
            if Path(args.outdir).exists():
                raise Exception(
                    "Output directory exists, use --overwrite if you want to overwrite it."
                )
        
        # 1 let's create a data table for all images.

        image_files = list()
        for f in Path(args.input).rglob("*.png"):
            try:
                img_metadata = get_metadata_from_img_filename(f)
                if args.input_mapping: #if input mapping passed as argument, it has priority
                    img_metadata['img_kmer_mapping'] = args.input_mapping
                if args.kmer_size: #if input kmer size passed as argument, it has priority
                    img_metadata['img_kmer_size'] = args.kmer_size


            except:
                img_metadata = {'sample': None,
                                'bp': None,
                                'img_kmer_mapping': args.input_mapping,
                                'img_kmer_size': args.kmer_size,
                                'path':f}


            if img_metadata['sample'] and img_metadata['bp']:
                fname = (
                         f"{img_metadata['sample']}{sample_bp_sep}"
                         f"{int(img_metadata['bp'] / 1000):08d}K{bp_kmer_sep}"
                         f"{args.output_mapping}{bp_kmer_sep}"
                         f"k{img_metadata['img_kmer_size']}.png"
                        )
                img_metadata['outfile_path'] = (Path(args.outdir)/
                                                Path(*img_metadata['path'].relative_to(Path(args.input)).parent.parts[1:])/
                                                fname)
            else:
                img_metadata['outfile_path'] = Path(args.outdir)/Path(*img_metadata['path'].parts[1:])
                
            image_files.append(img_metadata)

        eprint(f"Found {len(image_files)} files to convert.")
        eprint(f"Converted images will be written to {args.outdir}")

        if args.n_threads > 1:
            with multiprocessing.Pool(args.n_threads) as pool:
                process_partial = partial(process_remapping, 
                                          output_mapping=args.output_mapping,
                                          sum_rc=args.sum_reverse_complements)
                results = list(tqdm(pool.imap(process_partial, image_files), total=len(image_files), desc="Processing images"))
        
        else:
            for f_data in tqdm(image_files, desc="Processing images"):
                process_remapping(f_data, args.output_mapping, args.sum_reverse_complements)




    #delete any temporary directory created during execution
    try: #this will cause an error for train and convert commands, so need to catch exception
        if not args.int_folder and inter_dir.is_dir(): 
            shutil.rmtree(inter_dir)
    except:
        pass
    eprint("DONE")


if __name__ == "__main__":
    main()
