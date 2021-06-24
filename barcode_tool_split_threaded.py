#!/usr/bin/env python 

#imports used for all commands
from pathlib import Path
from collections import OrderedDict, defaultdict
from io import StringIO
from PIL import Image
import pandas as pd, numpy as np, tempfile, shutil, subprocess, functools
import re, sys, gzip, time, humanfriendly, random, multiprocessing, math


from fastai.torch_core import set_seed


# defining a function to print to stderr more easily
# idea from https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


#cleans illumina reads and saves results as a single merged fastq file to outpath
#cleaning includes:
#1 - adapter removal
#2 - deduplication
#3 - merging
def clean_reads(infiles, 
                outpath, 
                cut_adapters = True, 
                merge_reads = True, 
                max_bp = None, 
                threads = 1
               ):
    
    start_time = time.time()
    
    # let's print a message to the user
    basename = Path(outpath).name.removesuffix(''.join(Path(outpath).suffixes))
    
    if cut_adapters and merge_reads:
        eprint('Preprocessing', basename, 'to remove duplicates, adapters and merge reads')
    elif not cut_adapters and merge_reads:
        eprint('Preprocessing', basename, 'to remove duplicates and merge reads')
    elif cut_adapters and not merge_reads:
        eprint('Preprocessing', basename, 'to remove duplicates and adapters')
    else:
        eprint('Preprocessing', basename, 'to remove duplicates')
  
    #let's start separating forward, reverse and unpaired reads
    r1_re = re.compile(r'_R1_|_R1\.|\.R1\.|\.1\.|_1\.|_1_')
    r2_re = re.compile(r'_R2_|_R2\.|\.R2\.|\.2\.|_2\.|_2_')
    
    reads = {'R1': [str(x) for x in infiles if r1_re.search(str(x)) is not None],
             'R2': [str(x) for x in infiles if r2_re.search(str(x)) is not None]}
    reads['unpaired'] = [str(x) for x in infiles if str(x) not in reads['R1'] + reads['R2']]
    
    #if print_infiles:
    #    eprint('Note: Input files identified as:')
    #    eprint('Paired forward reads:')
    #    eprint('\n'.join(reads['R1']))
    #    eprint('Paired reverse reads:')
    #    eprint('\n'.join(reads['R2']))
    #    eprint('Unpaired reads:')
    #    eprint('\n'.join(reads['unpaired']))
    
    #from now on we will manipulate files in a temporary folder that will be deleted at the end of this function
    with tempfile.TemporaryDirectory(prefix='barcoding_clean_' + basename) as work_dir:
    #work_dir = tempfile.mkdtemp(prefix='barcoding_clean_' + basename)
        if True:

            # first, concatenate forward, reverse and unpaired reads
            # we will also store the number of basepairs for statistics
            initial_bp = 0
            retained_bp = 0
            write_out = True
            for k, readfiles in reads.items():
                if len(readfiles):
                    with open(Path(work_dir)/(basename + '_' + k +  '.fq'), 'wb+') as outfile:
                        for readf in sorted(readfiles):
                            if readf.endswith('gz'):
                                infile = gzip.open(readf, 'rb')
                            else:
                                infile = open(readf, 'rb')

                    
                            for line_n, line in enumerate(infile):
                                if line_n % 4 == 1:
                                    initial_bp += (len(line) - 1)
                                #if we reach a bp count 3 times as large as max_bp,
                                #this should be sufficient even after deduplication
                                #cleaning, etc
                                #so we stop writing out to avoid overhead
                                if ((line_n % 4 == 0) and 
                                     (write_out is True) and 
                                     (max_bp is not None) and 
                                     (initial_bp > (3 * max_bp))):
                                    retained_bp = initial_bp
                                    write_out = False
                                if write_out:
                                    outfile.write(line)

                            infile.close()
            if write_out:
                retained_bp = initial_bp

            # now we will use bbtools to deduplicate reads
            # here we will only deduplicate identical reads which is faster and should apply to most cases
            #deduplicate paired reads:
            if len(reads['R1']):
                command = ['clumpify.sh',
                           'in1=' + str(Path(work_dir)/(basename + '_R1.fq')),
                           'in2=' + str(Path(work_dir)/(basename + '_R2.fq')),
                           'out=' + str(Path(work_dir)/(basename + '_dedup_paired.fq')),
                           'dedupe',
                           'dupesubs=2',
                           'shortname=t',
                           'quantize=t',
                           '-Xmx1g',
                           'usetmpdir=t',
                           'tmpdir=' + str(Path(work_dir))]
                subprocess.run(command,
                               stderr = subprocess.DEVNULL,
                               stdout = subprocess.DEVNULL,
                               check = True)
                (Path(work_dir)/(basename + '_R1.fq')).unlink(missing_ok = True)
                (Path(work_dir)/(basename + '_R2.fq')).unlink(missing_ok = True)

            #deduplicate unpaired reads:
            if len(reads['unpaired']):
                command = ['clumpify.sh',
                           'interleaved=f',
                           'in=' + str(Path(work_dir)/(basename + '_unpaired.fq')),
                           'out=' + str(Path(work_dir)/(basename + '_dedup_unpaired.fq')),
                           'dedupe',
                           'dupesubs=2',
                           'shortname=t',
                           'quantize=t',
                           '-Xmx1g',
                           'usetmpdir=t',
                           'tmpdir=' + str(Path(work_dir))]
                subprocess.run(command,
                               stderr = subprocess.DEVNULL,
                               stdout = subprocess.DEVNULL, 
                               check = True)
                (Path(work_dir)/(basename + '_unpaired.fq')).unlink(missing_ok = True)
            
            #record statistics
            deduplicated = Path(work_dir).glob('*_dedup_*.fq')
            dedup_bp = 0
            for dedup in deduplicated:
                with open(dedup, 'rb') as infile:
                    BUF_SIZE = 100000000
                    tmp_lines = infile.readlines(BUF_SIZE)
                    line_n = 0        
                    while tmp_lines:
                        for line in tmp_lines:
                            if line_n % 4 == 1:
                                dedup_bp += (len(line) - 1)
                            line_n += 1
                        tmp_lines = infile.readlines(BUF_SIZE)


            # since fastp always writes a report file, we will redirect them to a temporary folder
            # that will be deleted
            with tempfile.TemporaryDirectory(prefix = 'fastp') as fastp_reports:
                # now we can run fastp to remove adapters and merge paired reads
                if (Path(work_dir)/(basename + '_dedup_paired.fq')).is_file():
                    # let's build the call to the subprocess. This is the common part
                    # for some reason, fastp fails with interleaved input unless it is provided from stdin
                    # for this reason, we will make a pipe
                    command = ['fastp',
                               '--stdin',
                               '--interleaved_in',
                               '--disable_quality_filtering',
                               '--disable_length_filtering',
                               '--thread', str(threads),
                               '--html', str(Path(fastp_reports)/'fastp.html'),
                               '--json', str(Path(fastp_reports)/'fastp.json'),
                               '--stdout'
                               ]

                    # add arguments depending on adapter option
                    if cut_adapters:
                        command.extend(['--detect_adapter_for_pe'])
                    else:
                        command.extend(['--disable_adapter_trimming'])

                    # add arguments depending on merge option
                    if merge_reads:
                        command.extend(['--merge',
                                        '--include_unmerged'])
                    else:
                        command.extend(['--stdout'])

                    with open(Path(work_dir)/(basename + '_clean_paired.fq'), 'wb') as outf:
                        cat = subprocess.Popen(['cat', str(Path(work_dir)/(basename + '_dedup_paired.fq'))],
                                              stdout = subprocess.PIPE)
                        subprocess.run(command,
                                       check= True,
                                       stdin = cat.stdout,
                                       stderr = subprocess.DEVNULL,
                                       stdout= outf)
                        (Path(work_dir)/(basename + '_dedup_paired.fq')).unlink(missing_ok = True)
                        


                # and remove adapters from unpaired reads, if any
                if (Path(work_dir)/(basename + '_dedup_unpaired.fq')).is_file():
                    command = ['fastp',
                               '-i', str(Path(work_dir)/(basename + '_dedup_unpaired.fq')),
                               '-o', str(Path(work_dir)/(basename + '_clean_unpaired.fq')),
                               '--html', str(Path(fastp_reports)/'fastp.html'),
                               '--json', str(Path(fastp_reports)/'fastp.json'),
                               '--disable_quality_filtering',
                               '--disable_length_filtering',
                               '--thread', str(threads)]

                    if not cut_adapters:
                        command.extend(['--disable_adapter_trimming'])

                    subprocess.run(command,
                                   stdout = subprocess.DEVNULL,
                                   stderr = subprocess.DEVNULL,
                                   check = True)
                    (Path(work_dir)/(basename + '_dedup_unpaired.fq')).unlink(missing_ok = True)


            # now that we finished cleaning, let's compress and move the final file to their destination folder
            # and assemble statistics
            # move files

            # create output folders if they do not exist

            try:
                outpath.parent.mkdir(parents = True)
            except OSError:
                pass

            command = ['cat']
            command.extend([str(x) for x in Path(work_dir).glob('*_clean_*.fq')])

            with open(outpath, 'wb') as outf:
                cat = subprocess.Popen(command, stdout = subprocess.PIPE)
                subprocess.run(['pigz', '-p', str(threads)], 
                               stdin = cat.stdout, 
                               stdout = outf,
                               stderr = subprocess.DEVNULL,
                               check = True)


            # stats: numbers of basepairs in each step (we recorded initial bp already when concatenating)
            clean = Path(work_dir).glob('*_clean_*.fq')
            if cut_adapters or merge_reads:
                clean_bp = 0
                for cl in clean:
                    with open(cl, 'rb') as infile:
                        BUF_SIZE = 100000000
                        tmp_lines = infile.readlines(BUF_SIZE)
                        line_n = 0        
                        while tmp_lines:
                            for line in tmp_lines:
                                if line_n % 4 == 1:
                                    clean_bp += (len(line) - 1)
                                line_n += 1
                            tmp_lines = infile.readlines(BUF_SIZE)
            else:
                clean_bp = np.nan
        
        
        stats = OrderedDict()        
        done_time = time.time()
        stats['start_basepairs'] = initial_bp
        stats['retained_basepairs'] = retained_bp
        stats['deduplicated_basepairs'] = dedup_bp
        stats['clean_basepairs'] = clean_bp
        stats['cleaning_time'] = done_time - start_time

    return(stats)


# function takes a fastq file as input and splits it into several files
# to be saved in outfolder with outprefix as prefix in the file name
# this is because we found that CNNs could accurately identify samples
# but below a certain level of coverage they were sensitive to amount
# of data used as input.
# Therefore, to better generalize we split the input files in several 
# The first one randomly pulls half of the reads, the next gets half of the remaining, etc
# until there are less than min_bp left
# if max_bp is set, the first file will use max_bp or half of the reads, whatever is lower
def split_fastq(infile, outprefix, outfolder, min_bp = 50000, max_bp = None, seed = None):
    start_time = time.time()
    
    if seed is not None:
        set_seed(seed)
    
    # let's start by counting the number of sites in reads of the input file
    sites_seq=[]
    with gzip.open(infile, 'rb') as f:
        for nlines,l in enumerate(f):
            if nlines % 4 == 1:
                sites_seq.append(len(l) - 1)
    nsites = sum(sites_seq)
    
    # now let's make a list that stores the number of sites we will retain in each file
    if max_bp is None:
        sites_per_file = [int(nsites)]
    elif int(nsites) > min_bp:
        sites_per_file = [min(int(nsites), max_bp)]
    else:
        raise Exception('Input file ' + 
                        infile + 
                        ' has less than ' + 
                        str(min_bp) + 
                        'bp, remove sample or raise min_bp.') 
    
    while sites_per_file[-1] > min_bp:
        oneless = sites_per_file[-1]-1
        nzeros = int(math.log10(oneless))
        first_digit = int(oneless/(10**nzeros))

        if first_digit in [1,2,5]:
            sites_per_file.append(first_digit*(10**(nzeros)))
        else:
            multiplier = max([x for x in [1,2,5] if x < first_digit])
            sites_per_file.append(multiplier*(10**(nzeros)))

    if sites_per_file[-1] < min_bp:
        del sites_per_file[-1]
        
    # now we will make a dictionary relating each read to its destination file
    # this will speed up sorting, since we will only read the input file once
    # and allocate each read to its appropriate output file
    seq_indices = [x for x in range(len(sites_seq))]
    destination = defaultdict(list)
    
    for ifile, nsites in enumerate(sites_per_file):
        idx = seq_indices.copy()
        random.shuffle(idx)
        sites_added = 0
        while sites_added < nsites:
            iseq = idx.pop()
            sites_added += sites_seq[iseq]
            destination[iseq].append(ifile)
                
    ##create output files
    Path(outfolder).mkdir(exist_ok = True)
    outfiles = [gzip.open(Path(outfolder)/(outprefix + '_' + str(int(bp/1000)).rjust(8, '0') + 'K' + '.fq.gz'),
                          'wb') 
                for bp in sites_per_file]

    # read input file and allocate reads to output files
    with gzip.open(infile, 'rb') as f:
        for i,l in enumerate(f):
            seq = i // 4 
            
            try:
                for dest in destination[seq]:
                    outfiles[dest].write(l)
            except KeyError: #some sequences will not be assigned to any file
                pass
            

    for f in outfiles:
        f.close()
            
    done_time = time.time()
    
    stats = OrderedDict()
    
    stats['splitting_time'] = done_time - start_time
    stats['splitting_bp_per_file'] = ','.join([str(x) for x in sites_per_file])
    
    return(stats)
            
#counts kmers for fastq infile, saving results as an hdf5 container
def count_kmers(infile, outfolder, threads = 1, k = 9):
    start_time = time.time()
    
    Path(outfolder).mkdir(exist_ok = True)
    outfile = Path(infile).name.removesuffix(''.join(Path(infile).suffixes)) + '.h5'
    subprocess.run(['dsk',
                    '-nb-cores', str(threads),
                    '-kmer-size', str(k),
                    '-abundance-min', '1',
                    '-abundance-min-threshold', '1',
                    '-max-memory', '1000',
                    '-file', str(infile),
                    '-out-dir', str(outfolder)
                   ],
                   stderr = subprocess.DEVNULL,
                   stdout = subprocess.DEVNULL,
                   check = True)
    
    done_time = time.time()
    
    stats = OrderedDict()
    stats['kmer_counting_time'] = done_time - start_time
    
    return(stats)


#given an input dsk hdf5 file, make an image with kmer counts
#mapping is the path to the table in parquet format that relates kmers to their positions in the image
#we do not need to save the ascii dsk kmer counts to disk, but the program requires a file
#so we will create a temporary file and delete it
def make_image(infile, outfolder, mapping, threads = 1):
    start_time = time.time()
    kmers = pd.read_parquet(mapping).set_index('canonical')
    
    Path(outfolder).mkdir(exist_ok = True)
    with tempfile.TemporaryDirectory(prefix='dsk') as outdir:
        # first, dump dsk results as ascii, save in a pandas df and merge with mapping
        dsk_out = subprocess.check_output(['dsk2ascii',
                                           '-c',
                                           '-file', str(infile),
                                           '-nb-cores', str(threads),
                                           '-out', str(Path(outdir)/'dsk.txt'),
                                           '-verbose', '0'
                                          ],
                                         stderr = subprocess.DEVNULL)
        dsk_out = dsk_out.decode('UTF-8')
        counts = pd.read_csv(StringIO(dsk_out), sep=" ",names=['sequence','count'],index_col=0)
        counts = kmers.join(counts)
        counts.loc[:,'count'] = counts['count'].fillna(0)
        
        #Now we will place counts in an array, log the counts and rescale to use 8 bit integers
        kmer_array = np.zeros(shape=[224,224,3])
        kmer_array
        kmer_array[224-counts['y'],counts['x'],counts['channel']-1] = np.float64(np.log10(counts['count']+1))
        kmer_array=np.uint8(kmer_array/np.max(kmer_array)*(2**8-1))
        
        #finally, save the image
        img = Image.fromarray(kmer_array)
        
        outfile = Path(infile).name.removesuffix(''.join(Path(infile).suffixes)) + '.png'
        img.save(Path(outfolder)/outfile)
    done_time = time.time()
    stats = OrderedDict()
    stats['img_time'] = done_time - start_time
    
    return(stats)
    

##train a cnn to recognize kmer images. input_table should have a column with species ID and another with associated image paths
##kwargs are optional keyword arguments passed to fine_tune()
##the final model will be saved to outfolder, as well as a text file listing the names of species in the order that they correspond to classes in the model
def train_cnn(input_table, outfolder, frozen_epochs, epochs):
    pass

    

if __name__ == "__main__":
    #imports for main program only
    import argparse
    
    # create top-level parser with common arguments
    main_parser = argparse.ArgumentParser(description = 'This program can be used to train or query a neural network to classify species based on barcode images produced from kmer frequencies in raw fastq files',
                                          add_help = True,
                                          formatter_class = argparse.ArgumentDefaultsHelpFormatter
                                         )
    subparsers = main_parser.add_subparsers(required = True, dest = 'command')
    
    parent_parser = argparse.ArgumentParser(add_help = False, 
                                            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parent_parser.add_argument('-M', '--max-bp' ,  help = 'maximum number of post-cleaning basepairs to include. Limiting the input data can speed up both training and queries but limit accuracy.')
    parent_parser.add_argument('-a', '--no-adapter', help = 'do not attempt to remove adapters from raw reads.', action='store_true')
    parent_parser.add_argument('-r', '--no-merge', help = 'do not attempt to merge paired reads.', action='store_true')
    parent_parser.add_argument('-i', '--int-folder', help = 'folder to write intermediate files (clean reads and images). If ommitted, only final result will be reported.')
    parent_parser.add_argument('-d', '--seed', help = 'random seed.')
    parent_parser.add_argument('-n', '--n-threads', help = 'number of samples to preprocess in parallel.', default = 2, type = int)
    parent_parser.add_argument('-c', '--cpus-per-thread', help = 'number of cpus to use for preprocessing each sample.', default = 2, type = int)

    
    # create parser for train command
    parser_train = subparsers.add_parser('train', parents = [parent_parser],
                                         formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                         help = 'Train a classifier based on raw reads. The program will automatically deduplicate, merge overlapping reads, clean adapters, count k-mers, produce images and train the neural network.')
    parser_train.add_argument('input', help = 'path to either the folder with fastq files or csv file relating file paths to samples. See online manual for formats.')
    parser_train.add_argument('-s', '--steps', help = 'steps to run: (1) clean and sort, (2) make image, (3) train model.', default = '123', type = str)
    parser_train.add_argument('-o', '--output', default = 'model', help = 'folder to write final fitted model')
    parser_train.add_argument('-m', '--min-bp' ,  type = str, help = 'minimum number of post-cleaning basepairs to make an image. Remaining reads below this threshold will be discarded', default = '10M')
    
    #create parser for query command
    parser_query = subparsers.add_parser('query', parents = [parent_parser],
                                         formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                         help = 'Query a fastq file agains a trained neural network. The program will automatically deduplicate, merge overlapping reads, clean adapters, count k-mers, produce images and query this image. The response is written to stdout in json format')

    parser_query.add_argument('model', help = 'pickle file with fitted neural network.')
    parser_query.add_argument('input', help = 'path to one or more fastq files to be queried.', nargs = '+')
    
    # execution
    args = main_parser.parse_args()
    
    # if no directory provided for intermediate results, create a temporary one 
    #  that will be deleted at the end of the program
    try:
        inter_dir = Path(args.int_folder)
    except TypeError:
        inter_dir = Path(tempfile.mkdtemp(prefix='barcoding_'))
        
        
    # set random seed if provided
    try:
        set_seed(args.seed)
    except TypeError:
        pass
    

    ###################
    # train command
    ###################
    
    if args.command == 'train':
        eprint('Training a new model')
        eprint('Reading input data')
        
        
        ##### STEP A - parse input and create a table relating reads files to samples and taxa
        inpath = Path(args.input)
        # first, check if input is a folder
        # if it is, make input table from the folder
        try:
            if inpath.is_dir():
                files_records = list()

                for taxon in inpath.iterdir():
                    if taxon.is_dir():
                        for sample in taxon.iterdir():
                            if sample.is_dir():
                                for fl in sample.iterdir():
                                    files_records.append({'taxon':taxon.name,
                                                          'sample':sample.name,
                                                          'reads_file': taxon/sample.name/fl.name
                                                         })

                files_table = pd.DataFrame(files_records)
                
                if not files_table.shape[0]:
                    raise Exception('Folder detected, but no records read. Check format.')

            # if it isn't a folder, read csv table input
            else:
                files_table = pd.read_csv(args.input)
                for colname in ['taxon', 'sample', 'reads_file']:
                    if colname not in files_table.columns:
                        raise Exception('Input csv file missing column: ' + colname)
                else:
                    files_table = files_table.assign(reads_file = lambda x: str(inpath.parent) + '/' + x['reads_file'])
        
        except:
            raise Exception('Could not parse input csv file or folder structure, double check.')
            
            
        ### Now we can run all steps, checking which were requested

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
            
        ### To be able to run samples in parallel using python, we will transform all the following code
        ### into a function
        ### this will take as input:
        ### one row of the condensed_files dataframe, the all_stats dict and number of cores per process

        def run_clean2img(it_row, cores_per_process = args.cpus_per_thread):
            x = it_row[1]
            stats = defaultdict(OrderedDict)
            
            clean_reads_f = inter_dir/'clean_reads'/(x['taxon'] + '__' + x['sample'] + '.fq.gz')
            split_reads_d = inter_dir/'split_fastqs'
            kmer_counts_d = inter_dir/'kmer_counts'
            images_d = inter_dir/'images'

            #### STEP B - clean reads and merge all files for each sample
            if '1' in args.steps:
                try:
                    maxbp = humanfriendly.parse_size(args.max_bp)
                except:
                    maxbp = None
                
                clean_stats = clean_reads(infiles = x['files'],
                            outpath = clean_reads_f,
                            cut_adapters = args.no_adapter is False,
                            merge_reads = args.no_merge is False,
                            max_bp = maxbp,
                            threads = cores_per_process
                           )

                stats[(x['taxon'],x['sample'])].update(clean_stats)

                #### STEP C - split clean reads into files with different number of reads
                eprint('Splitting fastqs for', x['taxon'] + '__' + x['sample'])
                split_stats = split_fastq(infile = clean_reads_f,
                            outprefix = (x['taxon'] + '__' + x['sample']),
                            outfolder = split_reads_d,
                            min_bp = humanfriendly.parse_size(args.min_bp), 
                            max_bp = maxbp, 
                            seed = None)

                stats[(x['taxon'],x['sample'])].update(split_stats)


            if '2' in args.steps:
                #### STEP D - count kmers 
                eprint('Creating images for', x['taxon'] + '__' + x['sample'])
                stats[(x['taxon'],x['sample'])]['kmer_counting_time'] = 0
                for infile in split_reads_d.glob(x['taxon'] + '__' + x['sample'] + '*'):
                    count_stats = count_kmers(infile = infile,
                                outfolder = kmer_counts_d, 
                                threads = cores_per_process,
                                k = 9)
                    stats[(x['taxon'],x['sample'])]['kmer_counting_time'] += count_stats['kmer_counting_time']


                #### STEP E - create images
                # the table mapping canonical kmers to pixels is stored as a feather file in
                # the same folder as this script
                map_table = Path(__file__).resolve().parent/'kmer_mapping.parquet'
                stats[(x['taxon'],x['sample'])]['img_time'] = 0
                for infile in kmer_counts_d.glob(x['taxon'] + '__' + x['sample'] + '*'):
                    img_stats = make_image(infile = infile, 
                                   outfolder = images_d, 
                                   mapping = map_table,
                                   threads = cores_per_process)
                    stats[(x['taxon'],x['sample'])]['img_time'] += img_stats['img_time']
            
            eprint('Images done for', x['taxon'] + '__' + x['sample'])
            return(stats)
        
        pool = multiprocessing.Pool(processes=int(args.n_threads))
        for stats in pool.imap_unordered(run_clean2img, condensed_files.iterrows(), chunksize = 1):
            all_stats.update(stats)
            (pd.DataFrame.from_dict(all_stats, orient = 'index').
              rename_axis(index=['taxon', 'sample']).
              to_csv('stats.csv')
            )
        pool.close()
        
        #for stats in res:
        #    all_stats.update(stats)
        
        




        
        # first, make a dictionary relating classes to samples and another relating samples to input files
    elif args.command == 'query':
        pass
    
    # if intermediate results were saved to a temporary file, delete them
    if not args.int_folder:
        shutil.rmtree(inter_dir)
        
    
    
    
    
    
