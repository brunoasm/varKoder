#!/usr/bin/env python 

##TO DO: implement option to overwrite or keep existing intermediate files

#import functions
from functions import *


#imports for main program only
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
parent_parser.add_argument('-n', '--n-threads', help = 'number of samples to preprocess in parallel.', default = 2, type = int)
parent_parser.add_argument('-c', '--cpus-per-thread', help = 'number of cpus to use for preprocessing each sample.', default = 2, type = int)
parent_parser.add_argument('-x', '--overwrite', help = 'overwrite existing results.', action='store_true')


# create parser for image command
parser_img = subparsers.add_parser('image', parents = [parent_parser],
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     help = 'Preprocess reads and prepare images for CNN training.')
parser_img.add_argument('input', help = 'path to either the folder with fastq files or csv file relating file paths to samples. See online manual for formats.')
parser_img.add_argument('-o','--outdir', help = 'path to folder where to write final images.', default = 'images')
parser_img.add_argument('-i', '--int-folder', help = 'folder to write intermediate files (clean reads and kmer counts). If ommitted, a temporary folder will be used.')
parser_img.add_argument('-m', '--min-bp' ,  type = str, help = 'minimum number of post-cleaning basepairs to make an image. Remaining reads below this threshold will be discarded', default = '10M')
parser_img.add_argument('-M', '--max-bp' ,  help = 'maximum number of post-cleaning basepairs to include.')
parser_img.add_argument('-a', '--no-adapter', help = 'do not attempt to remove adapters from raw reads.', action='store_true')
parser_img.add_argument('-r', '--no-merge', help = 'do not attempt to merge paired reads.', action='store_true')
parser_img.add_argument('-X', '--no-image', help = 'clean and split raw reads, but do not generate image.', action='store_true')
parser_img.add_argument('-k', '--kmer-size', help = 'size of kmers to count (5–8)', type = int, default = 7)

# create parser for train command
parser_train = subparsers.add_parser('train', parents = [parent_parser],
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     help = 'Train a CNN based on provided images.')
parser_train.add_argument('input', help = 'path to the folder with input images')
parser_train.add_argument('outdir', help = 'path to the folder where trained model will be stored')


#create parser for query command
parser_query = subparsers.add_parser('query', parents = [parent_parser],
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     help = 'Query a fastq file agains a trained neural network. The program will automatically deduplicate, merge overlapping reads, clean adapters, count k-mers, produce images and query this image. The response is written to stdout in json format')

parser_query.add_argument('model', help = 'pickle file with fitted neural network.')
parser_query.add_argument('input', help = 'path to one or more fastq files to be queried.', nargs = '+')
parser_query.add_argument('-a', '--no-adapter', help = 'do not attempt to remove adapters from raw reads.', action='store_true')
parser_query.add_argument('-r', '--no-merge', help = 'do not attempt to merge paired reads.', action='store_true')
parser_query.add_argument('-b', '--bp' ,  help = 'Number of post-cleaning basepairs to use for making image.')

# execution
args = main_parser.parse_args()

if args.kmer_size not in range(5, 9 + 1):
    raise Exception('kmer size must be between 5 and 9')

# if no directory provided for intermediate results, create a temporary one 
#  that will be deleted at the end of the program
try:
    inter_dir = Path(args.int_folder)
except TypeError:
    inter_dir = Path(tempfile.mkdtemp(prefix='barcoding_'))


# set random seed if provided
try:
    set_seed(args.seed)
    np_rng = np.random.default_rng(seed)
except TypeError:
    np_rng = np.random.default_rng()


###################
# image command
###################

if args.command == 'image':
    eprint('Kmer size:',str(args.kmer_size))
    eprint('Processing reads and preparing images')
    eprint('Reading input data')


    ##### STEP A - parse input and create a table relating reads files to samples and taxa
    inpath = Path(args.input)
    files_table = process_input(inpath)
 
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
    if Path('stats.csv').exists():
        all_stats.update(pd.read_csv('stats.csv', index_col = [0,1]).to_dict(orient = 'index'))
        
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

        clean_reads_f = inter_dir/'clean_reads'/(x['taxon'] + '__' + x['sample'] + '.fq.gz')
        split_reads_d = inter_dir/'split_fastqs'
        kmer_counts_d = inter_dir/(str(args.kmer_size) + 'mer_counts')
        images_d = Path(args.outdir)

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
                                  overwrite = args.overwrite
                   )

        stats[(x['taxon'],x['sample'])].update(clean_stats)

        #### STEP C - split clean reads into files with different number of reads
        eprint('Splitting fastqs for', x['taxon'] + '__' + x['sample'])
        split_stats = split_fastq(infile = clean_reads_f,
                                  outprefix = (x['taxon'] + '__' + x['sample']),
                                  outfolder = split_reads_d,
                                  min_bp = humanfriendly.parse_size(args.min_bp), 
                                  max_bp = maxbp, 
                                  overwrite = args.overwrite,
                                  seed = str(it_row[0]) + str(np_rng.integers(low = 0, high = 2**32)))

        stats[(x['taxon'],x['sample'])].update(split_stats)



        #### STEP D - count kmers 
        eprint('Creating images for', x['taxon'] + '__' + x['sample'])
        stats[(x['taxon'],x['sample'])][str(args.kmer_size) + 'mer_counting_time'] = 0
        
        kmer_key = str(args.kmer_size) + 'mer_counting_time'
        for infile in split_reads_d.glob(x['taxon'] + '__' + x['sample'] + '*'):
            count_stats = count_kmers(infile = infile,
                                      outfolder = kmer_counts_d, 
                                      threads = cores_per_process,
                                      k = args.kmer_size,
                                      overwrite = args.overwrite)
            
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
        for infile in kmer_counts_d.glob(x['taxon'] + '__' + x['sample'] + '*'):
            img_stats = make_image(infile = infile, 
                                   outfolder = images_d, 
                                   kmers = kmer_mapping,
                                   overwrite = args.overwrite,
                                   threads = cores_per_process)
            try:
                stats[(x['taxon'],x['sample'])][img_key] += img_stats[img_key]
            except KeyError as e:
                if e.args[0] == img_key:
                    pass
                else: 
                    raise(e)

        eprint('Images done for', x['taxon'] + '__' + x['sample'])
        return(stats)

    pool = multiprocessing.Pool(processes=int(args.n_threads))
    
    #for stats in pool.imap_unordered(run_clean2img, condensed_files.iterrows(), chunksize = 1):
    for stats in pool.imap_unordered(run_clean2img, condensed_files.iterrows(), chunksize = int(max(1, len(condensed_files.index)/args.n_threads/2))):
        try:
            all_stats.update(pd.read_csv('stats.csv', index_col = [0,1]).to_dict(orient = 'index'))
        except:
            pass
        
        for k in stats.keys():
            all_stats[k].update(stats[k])
        (pd.DataFrame.from_dict(all_stats, orient = 'index').
          rename_axis(index=['taxon', 'sample']).
          to_csv('stats.csv')
        )
    pool.close()

    #for stats in res:
    #    all_stats.update(stats)



###################
# query command
###################

elif args.command == 'query':
    pass

###################
# train command
###################


elif ags.command == 'train':
    pass



# if intermediate results were saved to a temporary file, delete them
if not args.int_folder:
    shutil.rmtree(inter_dir)

eprint('DONE')    
    
    
    
    
