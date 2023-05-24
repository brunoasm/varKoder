#!/usr/bin/env python
# -*- coding: utf-8 -*-


#import functions and libraries
from functions import *
import argparse

# create top-level parser with common arguments
main_parser = argparse.ArgumentParser(description = 'varKoder: using neural networks for DNA barcoding based on variation in whole-genome kmer frequencies',
                                      add_help = True,
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter
                                     )
subparsers = main_parser.add_subparsers(required = True, dest = 'command')

parent_parser = argparse.ArgumentParser(add_help = False, 
                                        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parent_parser.add_argument('-R', '--seed', help = 'random seed.', type = int)
parent_parser.add_argument('-x', '--overwrite', help = 'overwrite existing results.', action='store_true')



# create parser for image command
parser_img = subparsers.add_parser('image', parents = [parent_parser],
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     help = 'Preprocess reads and prepare images for CNN training.')
parser_img.add_argument('-v', '--verbose', 
                           help = 'show output for fastp, dsk and bbtools.',
                           action = 'store_true',
                           default = False
                          )
parser_img.add_argument('input', 
                        help = 'path to either the folder with fastq files or csv file relating file paths to samples. See online manual for formats.')
parser_img.add_argument('-k', '--kmer-size',
                        help = 'size of kmers to count (5–9)',
                        type = int, 
                        default = 7)
parser_img.add_argument('-n', '--n-threads', 
                        help = 'number of samples to preprocess in parallel.', 
                        default = 1, 
                        type = int)
parser_img.add_argument('-c', '--cpus-per-thread', 
                        help = 'number of cpus to use for preprocessing each sample.', 
                        default = 1, 
                        type = int)
parser_img.add_argument('-o','--outdir', 
                        help = 'path to folder where to write final images.', default = 'images')
parser_img.add_argument('-f', '--stats-file', 
                        help = 'path to file where sample statistics will be saved.', 
                        default ='stats.csv')
parser_img.add_argument('-i', '--int-folder', 
                        help = 'folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used.')
parser_img.add_argument('-m', '--min-bp',  
                        type = str, 
                        help = 'minimum number of post-cleaning basepairs to make an image. Samples below this threshold will be discarded', 
                        default = '500K')
parser_img.add_argument('-M', '--max-bp' ,  
                        help = 'maximum number of post-cleaning basepairs to make an image.')
parser_img.add_argument('-t', '--label-table', 
                        help = 'output a table with labels associated with each image, in addition to including them in the EXIF data.', 
                        action='store_true')
parser_img.add_argument('-a', '--no-adapter', 
                        help = 'do not attempt to remove adapters from raw reads.', 
                        action='store_true')
parser_img.add_argument('-r', '--no-merge', 
                        help = 'do not attempt to merge paired reads.', 
                        action='store_true')
parser_img.add_argument('-X', '--no-image', 
                        help = 'clean and split raw reads, but do not generate image.', 
                        action='store_true')



# create parser for train command
parser_train = subparsers.add_parser('train', parents = [parent_parser],
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     help = 'Train a CNN based on provided images.')
parser_train.add_argument('input', 
                          help = 'path to the folder with input images.')
parser_train.add_argument('outdir', 
                          help = 'path to the folder where trained model will be stored.')
parser_train.add_argument('-t','--label-table-path', 
                          help = 'path to csv table with labels for each sample. By default, varKoder will instead read labels from the image file metadata.')
parser_train.add_argument('-n','--single-label', 
                          help = 'Train as a single-label image classification model instead of multilabel. If multiple labels are provided, they will be concatenated to a single label.',
                          action = 'store_true'
                         )
parser_train.add_argument('-d','--threshold', 
                          help = 'Threshold to calculate precision and recall during for the validation set. Ignored if using --single-label',
                          type = float,
                          default = 0.7
                         )
parser_train.add_argument('-V','--validation-set',
                          help = 'comma-separated list of sample IDs to be included in the validation set. If not provided, a random validation set will be created. Turns off --validation-set-fraction'
                         ) 
parser_train.add_argument('-f','--validation-set-fraction',
                          help = 'fraction of samples to be held as a random validation set. Will be ignored if --validation-set is provided.',
                          type = float,
                          default = 0.2
                         ) 
parser_train.add_argument('-c','--architecture', 
                          help = 'model architecture to download from timm library. See https://github.com/rwightman/pytorch-image-models for possible options.',
                          default = 'vit_large_patch32_224'
                         )
parser_train.add_argument('-m','--pretrained-model', 
                          help = 'optional pickle file with pretrained model to update with new images. Turns off --architecture if used.'
                         )
parser_train.add_argument('-b', '--max-batch-size', 
                           help = 'maximum batch size when using GPU for training.',
                           type = int,
                           default=64 )
parser_train.add_argument('-r', '--base-learning-rate', 
                           help = 'base learning rate used in training. See https://walkwithfastai.com/lr_finder for information on learning rates.',
                           type = float,
                           default = 5e-3)
parser_train.add_argument('-e','--epochs', 
                          help = 'number of epochs to train. See https://docs.fast.ai/callback.schedule.html#learner.fine_tune',
                          type = int,
                          default = 17
                         )
parser_train.add_argument('-z','--freeze-epochs', 
                          help = 'number of freeze epochs to train. Recommended if using a pretrained model. See https://docs.fast.ai/callback.schedule.html#learner.fine_tune',
                          type = int,
                          default = 3
                         )
parser_train.add_argument('-P','--pretrained-timm', 
                          help = 'download pretrained model weights from timm. See https://github.com/rwightman/pytorch-image-models.',
                          action='store_true'
                         )
parser_train.add_argument('-i','--negative_downweighting', 
                          type = float,
                          default = 4,
                          help = 'Parameter controlling strength of loss downweighting for negative samples. See gamma(negative) parameter in https://arxiv.org/abs/2009.14119. Ignored if used with --single-label.',
                         )
#parser_train.add_argument('-i','--downweight-quality', 
#                          help = 'use a modified loss function that downweigths samples based on DNA quality. Ignored if used with --single-label.',
#                          action = 'store_true'
#                         )
parser_train.add_argument('-X','--mix-augmentation', 
                          help = 'apply MixUp or CutMix augmentation. See https://docs.fast.ai/callback.mixup.html',
                          choices=['CutMix', 'MixUp', 'None'],
                          default = 'MixUp'
                         )
parser_train.add_argument('-s','--label-smoothing', 
                          help = 'turn on Label Smoothing. Only used with --single-label. See https://github.com/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb',
                          action='store_true',
                          default = False
                         )
parser_train.add_argument('-p','--p-lighting', 
                          help = 'probability of a lighting transform. Set to 0 for no lighting transforms. See https://docs.fast.ai/vision.augment.html#aug_transforms',
                          type=float, 
                          default = 0.75
                         )
parser_train.add_argument('-l','--max-lighting', 
                          help = 'maximum scale of lighting transform. See https://docs.fast.ai/vision.augment.html#aug_transforms',
                          type=float, 
                          default = 0.25
                         )
parser_train.add_argument('-g', '--no-logging', 
                           help = 'hide fastai progress bar and logging during training.',
                           action = 'store_true',
                           default = False
                          )




#create parser for query command
parser_query = subparsers.add_parser('query', parents = [parent_parser],
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     help = 'Query a fastq file agains a trained neural network. The program will automatically deduplicate, merge overlapping reads, clean adapters, count k-mers, produce images and query this image. The response is written to stdout in json format')

parser_query.add_argument('model', 
                          help = 'pickle file with exported trained model.')
parser_query.add_argument('input', 
                          help = 'path to folder with fastq files to be queried.')
parser_query.add_argument('outdir', 
                          help = 'path to the folder where results will be saved.')
parser_query.add_argument('-v', '--verbose', 
                           help = 'show output for fastp, dsk and bbtools.',
                           action = 'store_true',
                           default = False
                          )
parser_query.add_argument('-I', '--images', 
                          help = 'input folder contains processed images instead of raw reads.', 
                          action = 'store_true')
parser_query.add_argument('-k', '--kmer-size', 
                          help = 'size of kmers to count (5–9)', 
                          type = int, 
                          default = 7)
parser_query.add_argument('-n', '--n-threads', 
                          help = 'number of samples to preprocess in parallel.', 
                          default = 1, 
                          type = int)
parser_query.add_argument('-c', '--cpus-per-thread', 
                          help = 'number of cpus to use for preprocessing each sample.', 
                          default = 1, 
                          type = int)
parser_query.add_argument('-f', '--stats-file', 
                          help = 'path to file where sample statistics will be saved.', 
                          default ='stats.csv')
parser_query.add_argument('-d','--threshold', 
                          help = 'confidence threshold to make a prediction.',
                          type=float, 
                          default = 0.7
                         )
parser_query.add_argument('-i', '--int-folder', 
                          help = 'folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used and deleted when done.')
parser_query.add_argument('-m','--keep-images', 
                          help = 'whether barcode images should be saved.',
                          action='store_true'
                         )
parser_query.add_argument('-a', '--no-adapter', 
                          help = 'do not attempt to remove adapters from raw reads.', 
                          action='store_true')
parser_query.add_argument('-r', '--no-merge', 
                          help = 'do not attempt to merge paired reads.', 
                          action='store_true')
parser_query.add_argument('-M', '--max-bp' ,  
                          help = 'number of post-cleaning basepairs to use for making image. If not provided, all data will be used.'
                         )
parser_query.add_argument('-b', '--max-batch-size', 
                           help = 'maximum batch size when using GPU for predictions.',
                           type = int,
                           default=64 )

# execution
args = main_parser.parse_args()


# set random seed
try:
    set_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
except TypeError:
    np_rng = np.random.default_rng()


##################################################
# preparing images for commands 'image' or 'query'
##################################################

if args.command == 'image' or (args.command == 'query' and not args.images):
         
    #check if kmer size provided is supported
    if args.kmer_size not in range(5, 9 + 1):
        raise Exception('kmer size must be between 5 and 9')

    # if no directory provided for intermediate results, create a temporary one 
    #  that will be deleted at the end of the program
    try:
        inter_dir = Path(args.int_folder)
    except TypeError:
        inter_dir = Path(tempfile.mkdtemp(prefix='barcoding_'))
        
    # set directory to save images
    if args.command == 'image':
        images_d = Path(args.outdir)
    elif args.command == 'query':
        if args.keep_images:
            images_d = Path(args.outdir)/'images'
            images_d.mkdir(parents=True, exist_ok=True)
        elif args.int_folder:
            images_d = Path(tempfile.mkdtemp(prefix='barcoding_img_'))
        else:
            images_d = inter_dir/'images'
    
    eprint('Kmer size:',str(args.kmer_size))
    eprint('Processing reads and preparing images')
    eprint('Reading input data')


    ##### STEP A - parse input and create a table relating reads files to samples and taxa
    inpath = Path(args.input)
    condensed_files = process_input(inpath, is_query = args.command == 'query')

    ### We will save statistics in a dictionary and then output as a table
    ### If a table already exists, read it
    ### If not, start from scratch
    all_stats = defaultdict(OrderedDict)
    stats_path = Path(args.stats_file)
    if stats_path.exists():
        all_stats.update(pd.read_csv(stats_path, index_col = [0]).to_dict(orient = 'index'))
        
    ### the same kmer mapping will be used for all files, so we will use it as a global variable
    map_path= Path(__file__).resolve().parent/'kmer_mapping'/(str(args.kmer_size) + 'mer_mapping.parquet')
    kmer_mapping = pd.read_parquet(map_path).set_index('kmer')

    # to use multiprocessing, we need to define the loop as a function, and avoid global variables
    # the partial function in functools does not work, so we do this by defining a new function:
    def partial_run_clean2img(it_row):
        return run_clean2img(it_row, 
                  kmer_mapping=kmer_mapping, 
                  args=args,
                  np_rng=np_rng,
                  inter_dir=inter_dir, 
                  all_stats=all_stats, 
                  stats_path=stats_path,
                  images_d=images_d,
                  label_sample_sep=label_sample_sep,
                  humanfriendly=humanfriendly,
                  defaultdict=defaultdict,
                  eprint=eprint,
                  make_image=make_image,
                  count_kmers=count_kmers
                 )
    
    
    #if only one sample processed at a time, we do a for loop
    if args.n_threads == 1:  
        for x in condensed_files.iterrows():
            stats = partial_run_clean2img(x)
            try:
                all_stats.update(read_stats(stats_path))
            except:
                pass
            for k in stats.keys():
                all_stats[k].update(stats[k])
            stats_df = stats_to_csv(all_stats, stats_path)
            
            if args.command == 'image' and args.label_table:
                (condensed_files.
                 merge(stats_df[['sample','base_frequencies_sd']].
                 assign(possible_low_quality = lambda x: x['base_frequencies_sd'] > qual_thresh)).
                 assign(labels= lambda x: x['labels'].apply(lambda y: labels_sep.join(y))).
                 loc[:,['sample', 'labels', 'possible_low_quality']].
                 to_csv(images_d/'labels.csv')
                )
        
    #if more than one sample processed at a time, use multiprocessing
    else:
        pool = multiprocessing.Pool(processes=int(args.n_threads))
    
        for stats in pool.imap_unordered(partial_run_clean2img, condensed_files.iterrows(), chunksize = int(max(1, len(condensed_files.index)/args.n_threads/2))):
        
            try:
                all_stats.update(read_stats(stats_path))
            except:
                pass
            
            for k in stats.keys():
                all_stats[k].update(stats[k])
                
            stats_df = stats_to_csv(all_stats, stats_path)
            
            if args.label_table:
                (condensed_files.
                 merge(stats_df[['sample','base_frequencies_sd']].
                 assign(possible_low_quality = lambda x: x['base_frequencies_sd'] > qual_thresh)).
                 assign(labels= lambda x: x['labels'].apply(lambda y: labels_sep.join(y))).
                 loc[:,['sample', 'labels', 'possible_low_quality']].
                 to_csv(images_d/'labels.csv')
                )
        
        pool.close()
    
    
    eprint('All images done, saved in',str(images_d))

    #for stats in res:
    #    all_stats.update(stats)
    
    
###################
# query command
###################

if args.command == 'query':
    if not args.overwrite:
        if Path(args.outdir,'predictions.csv').is_file():
            raise Exception('Output predictions file exists, use --overwrite if you want to overwrite it.')
            
        
    
    if args.images:
        images_d = Path(args.input)
    #if we provided sequences rather than images, they were processed in the command above
    

    img_paths = [img for img in images_d.iterdir() if img.name.endswith('png')]
    
    
    actual_labels = []
    for p in img_paths:
        try:
            labs = ';'.join(get_varKoder_labels(p))
        except AttributeError:
            labs = np.nan
            
        try:
            qual_flag = get_varKoder_qual(p)
        except AttributeError:
            qual_flag = np.nan
            
        try:
            freq_sd = get_varKoder_freqsd(p)
        except AttributeError:
            freq_sd = np.nan
            
        actual_labels.append(labs)
        
    n_images = len(img_paths)
    
    if n_images >= 128:
        eprint(n_images,'images in the input, will try to use GPU for prediction.')
        learn = load_learner(args.model, cpu = False)
    else:
        eprint(n_images,'images in the input, will use CPU for prediction.')
        learn = load_learner(args.model, cpu = True)
        
    df = pd.DataFrame({'path':img_paths})
    query_dl = learn.dls.test_dl(df,bs=args.max_batch_size)
    

        
    if 'MultiLabel' in str(learn.loss_func):
        eprint('This is a multilabel classification model, each input may have 0 or more predictions.')
        pp, _ = learn.get_preds(dl = query_dl, act = nn.Sigmoid())
        predictions_df = pd.DataFrame(pp)
        predictions_df.columns = learn.dls.vocab
        
        
        predicted_labels = predictions_df.apply(lambda row: ';'.join([col 
                                                                      for col in row.index
                                                                      if row[col] >= args.threshold]), 
                                                axis=1)
        
        
        predictions_df = pd.concat([pd.DataFrame({'varKode_image_path': img_paths,
                                          'sample_id':[(img.with_suffix('').
                                                       name.split(sample_bp_sep)[0].
                                                       split(label_sample_sep)[-1]) 
                                                      for img in img_paths],
                                          'query_basepairs':[(img.with_suffix('').
                                                             name.split(sample_bp_sep)[-1].
                                                             split(bp_kmer_sep)[0]) 
                                                            for img in img_paths],
                                          'query_kmer_len':[(img.with_suffix('').
                                                             name.split(sample_bp_sep)[-1].
                                                             split(bp_kmer_sep)[-1]) 
                                                            for img in img_paths],
                                          'trained_model_path':str(args.model),
                                          'prediction_type':'Multilabel',
                                          'prediction_threshold':args.threshold,
                                          'predicted_labels': predicted_labels,
                                          'actual_labels': actual_labels,
                                          'possible_low_quality': qual_flag,
                                          'basefrequency_sd': freq_sd
                                                   }),
                           predictions_df], axis = 1)
    else:
        eprint('This is a single label classification model, each input may will have only one prediction.')
        pp, _ = learn.get_preds(dl = query_dl)
        predictions_df = pd.DataFrame(pp)
        predictions_df.columns = learn.dls.vocab
        
        best_ps, best_idx = torch.max(pp, dim = 1)
        best_labels = learn.dls.vocab[best_idx]

        
        
        predictions_df = pd.concat([pd.DataFrame({'varKode_image_path': img_paths,
                                                  'sample_id':[(img.with_suffix('').
                                                               name.split(sample_bp_sep)[0].
                                                               split(label_sample_sep)[-1]) 
                                                              for img in img_paths],
                                                  'query_basepairs':[(img.with_suffix('').
                                                                     name.split(sample_bp_sep)[-1].
                                                                     split(bp_kmer_sep)[0]) 
                                                                    for img in img_paths],
                                                  'query_kmer_len':[(img.with_suffix('').
                                                                     name.split(sample_bp_sep)[-1].
                                                                     split(bp_kmer_sep)[-1]) 
                                                                    for img in img_paths],
                                                  'trained_model_path':str(args.model),
                                                  'prediction_type':'Single label',
                                                  'best_pred_label': best_labels,
                                                  'actual_labels': actual_labels,
                                                  'possible_low_quality': qual_flag,
                                                  'basefrequency_sd': freq_sd,
                                                  'best_pred_prob': best_ps
                                                           }),
                                   predictions_df], axis = 1)
    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)
    predictions_df.to_csv(outdir/'predictions.csv')
    
        
   



###################
# train command
###################


if args.command == 'train':


    eprint('Starting train command.')
    if not args.overwrite:
        if Path(args.outdir).exists():
            raise Exception('Output directory exists, use --overwrite if you want to overwrite it.')
        
    
    #1 let's create a data table for all images.
    image_files = list()
    for f in Path(args.input).rglob('*'):
        if f.name.endswith('png'):
            image_files.append({'sample':f.name.split(sample_bp_sep)[0],
                                'bp':int(f.name.split(sample_bp_sep)[1].split(bp_kmer_sep)[0].split('K')[0])*1000,
                                'path':f
                               })
    if args.label_table_path:
        image_files = (pd.DataFrame(image_files).
                       merge(pd.read_csv(args.label_table_path)[['sample','labels','possible_low_quality']], on = 'sample', how = 'left')
                      )
    else:
        image_files = (pd.DataFrame(image_files).
                       assign(labels = lambda x: x['path'].apply(lambda y: ';'.join(get_varKoder_labels(y))),
                              possible_low_quality = lambda x: x['path'].apply(get_varKoder_qual),
                             )
                      )
        
    #add quality-based sample weigths
    #if args.downweight_quality:
    #    image_files = image_files.assign(
    #            sample_weights = lambda x: x['path'].apply(get_varKoder_quality_weigths)
    #        )
    #else:
    #    image_files['sample_weights'] = 1

    
    #2 let's add a column to mark images in the validation set according to input options
    if args.validation_set: #if a specific validation set was defined, let's use it
        eprint('Spliting validation set as defined by user.')
        validation_samples = args.validation_set.split(',')
    else:
        eprint('Splitting validation set randomly. Fraction of samples per label combination held as validation:', str(args.validation_set_fraction))
        validation_samples = (image_files[['sample','labels']].
                              assign(labels = lambda x: x['labels'].apply(lambda y: ';'.join(sorted([z for z in y.split(';')])))).
                              drop_duplicates().
                              groupby('labels').
                              sample(frac = args.validation_set_fraction).
                              loc[:,'sample']
                             )
        
    image_files = image_files.assign(is_valid = image_files['sample'].isin(validation_samples),
                                     labels = lambda x: x['labels'].apply(lambda y: ';'.join(sorted([z for z in y.split(';')])))
                                    )
    
    #3 prepare input to training function based on options
    eprint('Setting up CNN model for training.')
    eprint('Model architecture:',args.architecture)
    
    callback = {'MixUp': MixUp,
                'CutMix': CutMix,
                'None': None}[args.mix_augmentation]


    trans = aug_transforms(do_flip = False,
                          max_rotate = 0,
                          max_zoom = 1,
                          max_lighting = args.max_lighting,
                          max_warp = 0,
                          p_affine = 0,
                          p_lighting = args.p_lighting
                          )


    #4 if a pretrained model has been provided, load model state
    load_on_cpu = True
    dev = torch.device('cpu')
    model_state_dict = None

    if torch.has_mps or (torch.has_cuda and torch.cuda.device_count()):
        load_on_cpu = False

    if args.pretrained_model:
        eprint("Loading pretrained model:", str(args.pretrained_model))
        past_learn = load_learner(args.pretrained_model, cpu = load_on_cpu)
        model_state_dict = past_learn.model.state_dict()
        pretrained = False
        del past_learn
    
    elif args.pretrained_timm:
        pretrained = True
        eprint("Starting model with pretrained weights from timm library.")
        
    else: 
        pretrained = False
        eprint("Starting model with random weigths.")
        
        
    if args.single_label:
        eprint('Single label model requested.')
        if (image_files['labels'].str.contains(';') == True).any():
            warnings.warn('Some samples contain more than one label. These will be concatenated. Maybe you want a multilabel model instead?', stacklevel=2)
            
        if args.mix_augmentation == 'None':
            loss = CrossEntropyLoss()
        else:
            loss = CrossEntropyLossFlat()

        eprint("Training for", args.freeze_epochs,"epochs with frozen model body weigths followed by", args.epochs,"epochs with unfrozen weigths.")
        #5 call training function
        learn = train_cnn(df = image_files, 
                          architecture = args.architecture, 
                          valid_pct = args.validation_set_fraction,
                          max_bs = args.max_batch_size,
                          base_lr = args.base_learning_rate,
                          epochs = args.epochs,
                          freeze_epochs = args.freeze_epochs,
                          normalize = True, 
                          pretrained = pretrained, 
                          callbacks = callback, 
                          max_lighting = args.max_lighting,
                          p_lighting = args.p_lighting,
                          loss_fn = loss,
                          model_state_dict = model_state_dict,
                          verbose = not args.no_logging
                         )
    else:
        eprint('Multilabel model requested.')
        if not (image_files['labels'].str.contains(';') == True).any():
            warnings.warn('No sample contains more than one label. Maybe you want a single label model instead?', stacklevel=2)
            
            
            
        eprint("Training for", args.freeze_epochs,"epochs with frozen model body weigths followed by", args.epochs,"epochs with unfrozen weigths.")
        #5 call training function        
        learn = train_multilabel_cnn(df = image_files, 
                  architecture = args.architecture, 
                  valid_pct = args.validation_set_fraction,
                  max_bs = args.max_batch_size,
                  base_lr = args.base_learning_rate,
                  epochs = args.epochs,
                  freeze_epochs = args.freeze_epochs,
                  normalize = True, 
                  pretrained = pretrained, 
                  callbacks = [callback], 
                  model_state_dict = model_state_dict,
                  metrics_threshold = args.threshold,
                  gamma_neg = args.negative_downweighting,
                  verbose = not args.no_logging,
                  max_lighting = args.max_lighting,
                  p_lighting = args.p_lighting
                 )
        

    # save results
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    learn.export(outdir/'trained_model.pkl')
    with open(outdir/'labels.txt','w') as outfile:
        outfile.write('\n'.join(learn.dls.vocab))
    image_files.to_csv(outdir/'input_data.csv')
        
    eprint('Model, labels, and data table saved to directory', str(outdir))
        
    
eprint('DONE')    
    
    
    
    
