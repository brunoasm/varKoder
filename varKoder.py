#!/usr/bin/env python
# -*- coding: utf-8 -*-


#import functions and libraries
from functions import *
import argparse

# create top-level parser with common arguments
main_parser = argparse.ArgumentParser(description = 'varKode: using neural networks for molecular barcoding based on variation in whole-genome kmer frequencies',
                                      add_help = True,
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter
                                     )
subparsers = main_parser.add_subparsers(required = True, dest = 'command')

parent_parser = argparse.ArgumentParser(add_help = False, 
                                        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parent_parser.add_argument('-d', '--seed', help = 'random seed.')
parent_parser.add_argument('-x', '--overwrite', help = 'overwrite existing results.', action='store_true')
parent_parser.add_argument('-b', '--max_batch_size', 
                           help = 'maximum batch size to use for training or querying.',
                           type = int,
                           default=64 )
parent_parser.add_argument('-v', '--verbose', 
                           help = 'show output for fastp, dsk and bbtools.',
                           action = 'store_true'
                          )



# create parser for image command
parser_img = subparsers.add_parser('image', parents = [parent_parser],
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     help = 'Preprocess reads and prepare images for CNN training.')
parser_img.add_argument('input', 
                        help = 'path to either the folder with fastq files or csv file relating file paths to samples. See online manual for formats.')
parser_img.add_argument('-k', '--kmer-size',
                        help = 'size of kmers to count (5–8)',
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
                        default = '10M')
parser_img.add_argument('-M', '--max-bp' ,  
                        help = 'maximum number of post-cleaning basepairs to make an image.')
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
parser_train.add_argument('-V','--validation-set',
                          help = 'comma-separated list of sample IDs to be included in the validation set. If not provided, a random validation set will be created.'
                         ) 
parser_train.add_argument('-f','--validation-set-fraction',
                          help = 'fraction of samples within each species to be held as a random validation set.',
                          type = float,
                          default = 0.2
                         ) 
parser_train.add_argument('-m','--pretrained-model', 
                          help = 'pickle file with optional pretrained model to update with new images.'
                         )
parser_train.add_argument('-e','--epochs', 
                          help = 'number of epochs to train. See https://docs.fast.ai/callback.schedule.html#learner.fine_tune',
                          type = int,
                          default = 20
                         )
parser_train.add_argument('-z','--freeze-epochs', 
                          help = 'number of freeze epochs to train. Recommended if using a pretrained model. See https://docs.fast.ai/callback.schedule.html#learner.fine_tune',
                          type = int,
                          default = 0
                         )
parser_train.add_argument('-r','--architecture', 
                          help = 'model architecture. See https://github.com/rwightman/pytorch-image-models for possible options.',
                          default = 'ig_resnext101_32x8d'
                         )
parser_train.add_argument('-X','--mix-augmentation', 
                          help = 'apply MixUp or CutMix augmentation. See https://docs.fast.ai/callback.mixup.html',
                          choices=['CutMix', 'MixUp', 'None'],
                          default = 'CutMix'
                         )
parser_train.add_argument('-s','--label-smoothing', 
                          help = 'turn on Label Smoothing. See https://github.com/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb',
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
                          default = 0.5
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
parser_query.add_argument('-I', '--images', 
                          help = 'input folder contains processed images instead of raw reads.', 
                          action = 'store_true')
parser_query.add_argument('-k', '--kmer-size', 
                          help = 'size of kmers to count (5–8)', 
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
                          help = 'Number of post-cleaning basepairs to use for making image. If not provided, all data will be used.'
                         )

# execution
args = main_parser.parse_args()


# set random seed
try:
    set_seed(args.seed)
    np_rng = np.random.default_rng(seed)
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
    files_table = process_input(inpath, is_query = args.command == 'query')
 
    condensed_files = (files_table.
                       groupby(['taxon','sample']).
                       agg({'reads_file': lambda x: list(x)}).
                       rename(columns = {'reads_file':'files'}).
                       reset_index()
                      )

    ### We will save statistics in a dictionary and then output as a table
    ### If a table already exists, read it
    ### If not, start from scratch
    all_stats = defaultdict(OrderedDict)
    stats_path = Path(args.stats_file)
    if stats_path.exists():
        all_stats.update(pd.read_csv(stats_path, index_col = [0,1]).to_dict(orient = 'index'))
        
    ### the same kmer mapping will be used for all files, so we will use it as a global variable
    map_path= Path(__file__).resolve().parent/'kmer_mapping'/(str(args.kmer_size) + 'mer_mapping.parquet')
    kmer_mapping = pd.read_parquet(map_path).set_index('kmer')

    ### To be able to run samples in parallel using python, we will transform all the following code
    ### into a function
    ### this will take as input:
    ### one row of the condensed_files dataframe, the all_stats dict and number of cores per process

    def run_clean2img(it_row, cores_per_process = args.cpus_per_thread):
        x = it_row[1]
        stats = defaultdict(OrderedDict)

        clean_reads_f = inter_dir/'clean_reads'/(x['taxon'] + label_sample_sep + x['sample'] + '.fq.gz')
        split_reads_d = inter_dir/'split_fastqs'
        kmer_counts_d = inter_dir/(str(args.kmer_size) + 'mer_counts')


        #### STEP B - clean reads and merge all files for each sample
        try:
            maxbp = int(humanfriendly.parse_size(args.max_bp))
        except:
            maxbp = None

        clean_stats = clean_reads(infiles = x['files'],
                                  outpath = clean_reads_f,
                                  cut_adapters = args.no_adapter is False,
                                  merge_reads = args.no_merge is False,
                                  max_bp = maxbp,
                                  threads = cores_per_process,
                                  overwrite = args.overwrite,
                                  verbose = args.verbose
                   )

        stats[(x['taxon'],x['sample'])].update(clean_stats)

        #### STEP C - split clean reads into files with different number of reads
        eprint('Splitting fastqs for', x['taxon'] + label_sample_sep + x['sample'])
        if args.command == 'image':
            split_stats = split_fastq(infile = clean_reads_f,
                                      outprefix = (x['taxon'] + 
                                                   label_sample_sep + 
                                                   x['sample']),
                                      outfolder = split_reads_d,
                                      min_bp = humanfriendly.parse_size(args.min_bp), 
                                      max_bp = maxbp, 
                                      overwrite = args.overwrite,
                                      verbose = args.verbose,
                                      seed = str(it_row[0]) + str(np_rng.integers(low = 0, high = 2**32)))
        elif args.command == 'query':
             split_stats = split_fastq(infile = clean_reads_f,
                                      outprefix = (x['taxon'] + 
                                                   label_sample_sep + 
                                                   x['sample']),
                                      outfolder = split_reads_d,
                                      is_query = True,
                                      max_bp = maxbp, 
                                      overwrite = args.overwrite,
                                      verbose = args.verbose,
                                      seed = str(it_row[0]) + str(np_rng.integers(low = 0, high = 2**32)))           

        stats[(x['taxon'],x['sample'])].update(split_stats)
        
        eprint('Cleaning and splitting reads done for', x['taxon'] + label_sample_sep + x['sample'])



        #### STEP D - count kmers
        if args.command == 'query' or not args.no_image:
            eprint('Counting kmers and creating images for', x['taxon'] + label_sample_sep + x['sample'])
            stats[(x['taxon'],x['sample'])][str(args.kmer_size) + 'mer_counting_time'] = 0

            kmer_key = str(args.kmer_size) + 'mer_counting_time'
            for infile in split_reads_d.glob(x['taxon'] + label_sample_sep + x['sample'] + '*'):
                count_stats = count_kmers(infile = infile,
                                          outfolder = kmer_counts_d, 
                                          threads = cores_per_process,
                                          k = args.kmer_size,
                                          overwrite = args.overwrite,
                                          verbose = args.verbose
                                         )

                try:
                    stats[(x['taxon'],x['sample'])][kmer_key] += count_stats[kmer_key]
                except KeyError as e:
                    if e.args[0] == kmer_key:
                        pass
                    else: 
                        raise(e)


            #### STEP E - create images
            # the table mapping canonical kmers to pixels is stored as a feather file in
            # the same folder as this script

            img_key = 'k' + str(args.kmer_size) + '_img_time'

            stats[(x['taxon'],x['sample'])][img_key] = 0
            for infile in kmer_counts_d.glob(x['taxon'] + label_sample_sep + x['sample'] + '*'):
                img_stats = make_image(infile = infile, 
                                       outfolder = images_d, 
                                       kmers = kmer_mapping,
                                       overwrite = args.overwrite,
                                       threads = cores_per_process,
                                       verbose = args.verbose
                                      )
                try:
                    stats[(x['taxon'],x['sample'])][img_key] += img_stats[img_key]
                except KeyError as e:
                    if e.args[0] == img_key:
                        pass
                    else: 
                        raise(e)

            eprint('Images done for', x['taxon'] + label_sample_sep + x['sample'])
        return(stats)
    
    
    #if only one sample processed at a time, we do a for loop
    if args.n_threads == 1:  
        for x in condensed_files.iterrows():
            stats = run_clean2img(x)
            try:
                all_stats.update(pd.read_csv(stats_path, index_col = [0,1]).to_dict(orient = 'index'))
            except:
                pass
            for k in stats.keys():
                all_stats[k].update(stats[k])
            (pd.DataFrame.from_dict(all_stats, orient = 'index').
              rename_axis(index=['taxon', 'sample']).
              to_csv(stats_path)
            )
        
    #if more than one sample processed at a time, use multiprocessing
    else:
        pool = multiprocessing.Pool(processes=int(args.n_threads))
    
        for stats in pool.imap_unordered(run_clean2img, condensed_files.iterrows(), chunksize = int(max(1, len(condensed_files.index)/args.n_threads/2))):
            try:
                all_stats.update(pd.read_csv(stats_path, index_col = [0,1]).to_dict(orient = 'index'))
            except:
                pass
            
            for k in stats.keys():
                all_stats[k].update(stats[k])
            (pd.DataFrame.from_dict(all_stats, orient = 'index').
              rename_axis(index=['taxon', 'sample']).
              to_csv(stats_path)
            )
        pool.close()
    
    
    eprint('All images done, saved in',str(images_d))

    #for stats in res:
    #    all_stats.update(stats)
    
    
###################
# query command
###################

if args.command == 'query':
    if args.images:
        images_d = Path(args.input)
    #if we provided sequences rather than images, they were processed in the command above
        
    img_paths = [img for img in images_d.iterdir() if img.name.endswith('png')]
        
    n_images = len(img_paths)
    
    if n_images >= 128:
        eprint(n_images,'images in the input, will try to use GPU for prediction.')
        learn = load_learner(args.model, cpu = False)
    else:
        eprint(n_images,'images in the input, will use CPU for prediction.')
        learn = load_learner(args.model, cpu = True)
    
    df = pd.DataFrame({'path':img_paths})
    query_dl = learn.dls.test_dl(df,bs=args.max_batch_size)
    pp, _ = learn.get_preds(dl = query_dl)
    best_ps, best_idx = torch.max(pp, dim = 1)
    best_labels = learn.dls.vocab[best_idx]
    predictions_df = pd.DataFrame(pp)
    predictions_df.columns = learn.dls.vocab
    
    predictions_df = pd.concat([pd.DataFrame({'sample_id':[(img.with_suffix('').
                                                           name.split(label_sample_sep)[-1].
                                                           split(sample_bp_sep)[0]) 
                                                          for img in img_paths],
                                              'varKode_image_path': img_paths, 
                                              'basepairs_used':[(img.with_suffix('').
                                                                 name.split(label_sample_sep)[-1].
                                                                 split(sample_bp_sep)[1]) 
                                                                for img in img_paths],
                                              'best_pred_label': best_labels,
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
    
    #1 let's create a data table for all images.
    image_files = list()
    for f in Path(args.input).rglob('*'):
        if str(f).endswith('png'):
            image_files.append({'taxon':f.name.split(label_sample_sep)[0],
                                'sample':f.name.split(label_sample_sep)[1].split(sample_bp_sep)[0],
                                'bp':int(f.name.split(label_sample_sep)[1].split(sample_bp_sep)[1].split('K')[0])*1000,
                                'path':f
                               })
    image_files = pd.DataFrame(image_files)
    
    #2 let's add a column to mark images in the validation set according to input options
    if args.validation_set: #if a specific validation set was defined, let's use it
        validation_samples = args.validation_set.split(',')
    else:
        validation_samples = (image_files[['taxon','sample']].
                              drop_duplicates().
                              groupby('taxon').
                              sample(frac = args.validation_set_fraction).
                              loc[:,'sample']
                             )
        
    image_files = image_files.assign(is_valid = image_files['sample'].isin(validation_samples))
    
        
    #3 prepare input to training function based on options
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
    
    if args.label_smoothing:
        if args.mix_augmentation == 'None':
            loss = LabelSmoothingCrossEntropy()
        else:
            loss = LabelSmoothingCrossEntropyFlat()
    else:
        if args.mix_augmentation == 'None':
            loss = CrossEntropyLoss()
        else:
            loss = CrossEntropyLossFlat()
    
    #4 if a pretrained model has been provided, load model state
    load_on_cpu = True
    model_state_dict = None
    
    try: #within a try-except statement since currently only nightly build has this function
        if torch.has_mps:
            load_on_cpu = False
    except AttributeError:
        pass
    
    if torch.has_cuda and torch.cuda.device_count():
        load_on_cpu = False
    
    if args.pretrained_model:
        past_learn = load_learner(args.pretrained_model, cpu = load_on_cpu)
        model_state_dict = past_learn.model.state_dict()
        del past_learn
    
    
    
    #5 call training function
    learn = train_cnn(image_files, 
                      args.architecture, 
                      max_bs = args.max_batch_size,
                      epochs = args.epochs,
                      freeze_epochs = args.freeze_epochs,
                      normalize = True, 
                      pretrained = False, 
                      callbacks = callback, 
                      transforms = trans,
                      loss_fn = loss,
                      model_state_dict = model_state_dict
                     )
    
    # save results
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    learn.export(outdir/'trained_model.pkl')
    with open(outdir/'labels.txt','w') as outfile:
        outfile.write('\n'.join(learn.dls.vocab))
    image_files.to_csv(outdir/'input_data.csv')
        
    eprint('Model, labels, and data table saved to directory', str(outdir))
        
    




# if intermediate results were saved to a temporary folder, delete them
#try:
#    if not args.int_folder:
#        shutil.rmtree(inter_dir)
#except AttributeError:
#    pass
#
#try:
#    if args.command == 'query' and not args.images and not args.keep_images:
#        shutil.rmtree(images_d)
#except AttributeError:
#    pass
#        

eprint('DONE')    
    
    
    
    
