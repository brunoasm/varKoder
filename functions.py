#!/usr/bin/env python 

#imports used for all commands
from pathlib import Path
from collections import OrderedDict, defaultdict
from io import StringIO
from PIL import Image
#from matplotlib import cm
from tenacity import Retrying, stop_after_attempt, wait_random_exponential
import pandas as pd, numpy as np, tempfile, shutil, subprocess, functools
import re, sys, gzip, time, humanfriendly, random, multiprocessing, math

#from scipy.stats import boxcox
#from scipy.stats import rankdata

from fastai.data.all import DataBlock, ColSplitter, ColReader, get_c
from fastai.vision.all import ImageBlock, CategoryBlock, create_head, apply_init, default_split
from fastai.vision.learner import _update_first_layer, has_pool_type
from fastai.vision.all import aug_transforms
from fastai.metrics import accuracy
from fastai.learner import Learner, load_learner
from fastai.torch_core import set_seed
from fastai.callback.mixup import CutMix, MixUp
from fastai.callback.hook import num_features_model
from fastai.losses import LabelSmoothingCrossEntropyFlat, CrossEntropyLossFlat, LabelSmoothingCrossEntropy
from torch.nn import CrossEntropyLoss
from torch import nn
import torch
from timm import create_model
from math import log

from fastai.torch_core import set_seed

#define filename separators
label_sample_sep = '+'
sample_bp_sep = '@'


# defining a function to print to stderr more easily
# idea from https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    

# this function process the input file or folder and returns a table with files
def process_input(inpath, is_query = False):
   # first, check if input is a folder
    # if it is, make input table from the folder
    try:
        if inpath.is_dir() and not is_query:
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
        
        elif is_query:
            files_records = list()
            #start by checking if input directory contains directories
            contains_dir = False
            for f in inpath.iterdir():
                if f.is_dir():
                    contains_dir = True
                    break
                    
            #if there are no subdirectories, treat everything as a single sample. Otherwise, use each directory for a sample
            if not contains_dir:
                for fl in inpath.iterdir():
                    if fl.name.endswith('fq') or fl.name.endswith('fastq') or fl.name.endswith('fq.gz') or fl.name.endswith('fastq.gz'):
                        files_records.append({'taxon':'query',
                                             'sample':inpath.name,
                                             'reads_file':fl})
                        
            else:
                for sample in inpath.iterdir():
                    if sample.is_dir():
                        for fl in sample.iterdir():
                            if fl.name.endswith('fq') or fl.name.endswith('fastq') or fl.name.endswith('fq.gz') or fl.name.endswith('fastq.gz'):
                                files_records.append({'taxon':'query',
                                                     'sample': sample.name,
                                                     'reads_file':sample/fl.name})

            

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
        
    return(files_table)


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
                threads = 1,
                overwrite = False):
    
    start_time = time.time()
    
    # let's print a message to the user
    basename = Path(outpath).name.removesuffix(''.join(Path(outpath).suffixes))

    if not overwrite and outpath.is_file():
        eprint('Skipping cleaning for', basename + ':', 'File exists.')
        return(OrderedDict())
    
    if cut_adapters and merge_reads:
        eprint('Preprocessing', basename, 'to remove duplicates, adapters and merge reads')
    elif not cut_adapters and merge_reads:
        eprint('Preprocessing', basename, 'to remove duplicates and merge reads')
    elif cut_adapters and not merge_reads:
        eprint('Preprocessing', basename, 'to remove duplicates and adapters')
    else:
        eprint('Preprocessing', basename, 'to remove duplicates')
        
  
    #let's start separating forward, reverse and unpaired reads
    re_pair = {'R1':re.compile(r'_R1_|_R1\.|\.R1\.|\.1\.|_1\.|_1_'),
              'R2':re.compile(r'_R2_|_R2\.|\.R2\.|\.2\.|_2\.|_2_')}
    
    reads = {'R1': [str(x) for x in infiles if re_pair['R1'].search(str(x)) is not None],
             'R2': [str(x) for x in infiles if re_pair['R2'].search(str(x)) is not None]}
    reads['unpaired'] = [str(x) for x in infiles if str(x) not in reads['R1'] + reads['R2']]
    
    #if print_infiles:
    #    eprint('Note: Input files identified as:')
    #    eprint('Paired forward reads:')
    #    eprint('\n'.join(reads['R1']))
    #    eprint('Paired reverse reads:')
    #    eprint('\n'.join(reads['R2']))
    #    eprint('Unpaired reads:')
    #    eprint('\n'.join(reads['unpaired']))
    
    #let's check if destination files exist, and skip them if we can
    
    #from now on we will manipulate files in a temporary folder that will be deleted at the end of this function
    #with tempfile.TemporaryDirectory(prefix='barcoding_clean_' + basename) as work_dir:
    work_dir = tempfile.mkdtemp(prefix='barcoding_clean_' + basename)
    #if True:

    # first, concatenate forward, reverse and unpaired reads
    # we will also store the number of basepairs for statistics
    initial_bp = {'unpaired':0, 'R1':0, 'R2':0}
    lines_concat = {'unpaired':dict(), 'R1':dict(), 'R2':dict()}
    
    write_out = True
    
    for k in ['unpaired','R1','R2']:
        readfiles = reads[k]
        
        if len(readfiles):
            with open(Path(work_dir)/(basename + '_' + k +  '.fq'), 'wb') as outfile:
                for i_f, readf in enumerate(sorted(readfiles)):
                    #initializes count of number of lines written out for this file
                    lines_concat[k][i_f] = 0
                    
                    #opens file
                    if readf.endswith('gz'):
                        infile = gzip.open(readf, 'rb')
                    else:
                        infile = open(readf, 'rb')


                    #reads file
                    for line_n, line in enumerate(infile):
                        if line_n % 4 == 1:
                            #counts number of lines in file originally
                            initial_bp[k] += (len(line) - 1)

                        #if we reach a bp count 5 times as large as max_bp,
                        #this should be sufficient even after deduplication,
                        #cleaning, etc
                        #so we stop writing out to decrease processing time
                        #in case this happens while reading R1, we will record how many lines we read 
                        #and read the same number of lines in corresponding R2
                        
                        if write_out is True:
                            bp_read = initial_bp['unpaired'] + 2 * initial_bp['R1']
                            if ((line_n % 4 == 0) and 
                                (max_bp is not None) and 
                                (bp_read > (5 * max_bp))):
                                write_out = False
                            else:
                                outfile.write(line)
                                lines_concat[k][i_f] = line_n

                        elif k == 'R2' and lines_concat[k][i_f] < lines_concat['R1'][i_f]:
                            bp_read = sum([v for k,v in initial_bp.items()])
                            outfile.write(line)
                            lines_concat[k][i_f] = line_n
                            
                    infile.close()
            

    if write_out: #if we reach the end and are still writing out, count how many bp retained
        bp_read = sum([v for k,v in initial_bp.items()])


    retained_bp = bp_read

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
        
        try:
            p = subprocess.run(command,
                           stderr = subprocess.PIPE,
                           stdout = subprocess.DEVNULL,
                           check = True)
            eprint(p.stderr.decode())
        except subprocess.CalledProcessError as e:
            eprint('clumpify.sh returned an error:')
            eprint(e.stderr.decode())
            eprint('we will now try to fix the input files with repair.sh')
            command = ['repair.sh',
                   'in1=' + str(Path(work_dir)/(basename + '_R1.fq')),
                   'in2=' + str(Path(work_dir)/(basename + '_R2.fq')),
                   'out=' + str(Path(work_dir)/(basename + '_fixed_paired.fq'))]
            eprint('input files fixed, resuming')
            p = subprocess.run(command,
                           stderr = subprocess.PIPE,
                           stdout = subprocess.DEVNULL,
                           check = True)
            eprint(p.stderr.decode())
            command = ['clumpify.sh',
                   'in=' + str(Path(work_dir)/(basename + '_fixed_paired.fq')),
                   'out=' + str(Path(work_dir)/(basename + '_dedup_paired.fq')),
                   'dedupe',
                   'dupesubs=2',
                   'shortname=t',
                   'quantize=t',
                   '-Xmx1g',
                   'usetmpdir=t',
                   'tmpdir=' + str(Path(work_dir))]
            
        (Path(work_dir)/(basename + '_R1.fq')).unlink(missing_ok = True)
        (Path(work_dir)/(basename + '_R2.fq')).unlink(missing_ok = True)
        (Path(work_dir)/(basename + '_fixed_paired.fq')).unlink(missing_ok = True)

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
        p = subprocess.run(command,
                       stderr = subprocess.PIPE,
                       stdout = subprocess.DEVNULL, 
                       check = True)
        eprint(p.stderr.decode())
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
                   '--trim_poly_g',
                   '--thread', str(threads),
                   '--html', str(Path(work_dir)/(basename + '_fastp_paired.html')),
                   '--json', str(Path(work_dir)/(basename + '_fastp_paired.json')),
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
            p = subprocess.run(command,
                           check= True,
                           stdin = cat.stdout,
                           stderr = subprocess.PIPE,
                           stdout= outf)
            eprint(p.stderr.decode())
            (Path(work_dir)/(basename + '_dedup_paired.fq')).unlink(missing_ok = True)



        # and remove adapters from unpaired reads, if any
        if (Path(work_dir)/(basename + '_dedup_unpaired.fq')).is_file():
            command = ['fastp',
                       '-i', str(Path(work_dir)/(basename + '_dedup_unpaired.fq')),
                       '-o', str(Path(work_dir)/(basename + '_clean_unpaired.fq')),
                       '--html', str(Path(work_dir)/(basename + '_fastp_unpaired.html')),
                       '--json', str(Path(work_dir)/(basename + '_fastp_unpaired.json')),
                       '--disable_quality_filtering',
                       '--disable_length_filtering',
                       '--trim_poly_g',
                       '--thread', str(threads)]

            if not cut_adapters:
                command.extend(['--disable_adapter_trimming'])

            p = subprocess.run(command,
                           stdout = subprocess.DEVNULL,
                           stderr = subprocess.PIPE,
                           check = True)
            eprint(p.stderr.decode())
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
        p = subprocess.run(['pigz', '-p', str(threads)], 
                       stdin = cat.stdout, 
                       stdout = outf,
                       stderr = subprocess.PIPE,
                       check = True)
        eprint(p.stderr.decode())

    #copy fastp reports
    for fastp_f in Path(work_dir).glob('*fastp*'):
        shutil.copy(fastp_f,outpath.parent/str(fastp_f.name))


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
    stats['deduplicated_basepairs'] = dedup_bp
    stats['clean_basepairs'] = clean_bp
    stats['cleaning_time'] = done_time - start_time
        
        
    shutil.rmtree(work_dir)
    ####END OF TEMPDIR BLOCK    

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
def split_fastq(infile, 
                outprefix, 
                outfolder, 
                min_bp = 50000, 
                max_bp = None,
                is_query = False,
                seed = None,
                overwrite = False):
    start_time = time.time()
        
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
    elif is_query or int(nsites) > min_bp:
        sites_per_file = [min(int(nsites), int(max_bp))]
    else:
        raise Exception('Input file ' + 
                        infile + 
                        ' has less than ' + 
                        str(min_bp) + 
                        'bp, remove sample or raise min_bp.') 
        
    if not is_query:
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
        
    # now we will use bbtools reformat.sh to subsample
    outfs = [Path(outfolder)/(outprefix + sample_bp_sep + str(int(bp/1000)).rjust(8, '0') + 'K' + '.fq.gz') for bp in sites_per_file]
    
    if all([f.is_file() for f in outfs]):
        if not overwrite:
            eprint('Files exist. Skipping subsampling for file:',str(infile))
            return(OrderedDict())
    
    for i, bp in enumerate(sites_per_file):
        outfile = Path(outfolder)/(outprefix + sample_bp_sep + str(int(bp/1000)).rjust(8, '0') + 'K' + '.fq.gz')
        
        subprocess.run(['reformat.sh',
                        'samplebasestarget=' + str(bp),
                        'sampleseed='+str(int(seed) + i),
                        'in=' + str(infile),
                        'out=' + str(outfile),
                        'overwrite=true'
                       ],
                      stderr = subprocess.DEVNULL,
                      stdout = subprocess.DEVNULL,
                      check = True)
            
    done_time = time.time()
    
    stats = OrderedDict()
    
    stats['splitting_time'] = done_time - start_time
    stats['splitting_bp_per_file'] = ','.join([str(x) for x in sites_per_file])
    
    return(stats)
            
#counts kmers for fastq infile, saving results as an hdf5 container
#in some cases dsk lauches an hdf5 error of file access when using more than 1 thread. For that reason, will keep one thread for now
def count_kmers(infile, 
                outfolder, 
                threads = 1, 
                k = 7,
                overwrite = False
               ):
    start_time = time.time()
    
    Path(outfolder).mkdir(exist_ok = True)
    outfile = Path(infile).name.removesuffix(''.join(Path(infile).suffixes)) + '.fq.h5'
    outpath = outfolder/outfile
    
    if not overwrite and outpath.is_file():
        eprint('File exists. Skipping kmer counting for file:',str(infile))
        return(OrderedDict())
    
    with tempfile.TemporaryDirectory(prefix='dsk') as tempdir:
        for attempt in Retrying(stop = stop_after_attempt(5), wait = wait_random_exponential(multiplier=1, max=60)):
            with attempt:
                subprocess.run(['dsk',
                                '-nb-cores', str(threads),
                                '-kmer-size', str(k),
                                '-abundance-min', '1',
                                '-abundance-min-threshold', '1',
                                '-max-memory', '1000',
                                '-file', str(infile),
                                '-out-tmp', str(tempdir),
                                '-out-dir', str(outfolder)
                               ],
                               stderr = subprocess.DEVNULL,
                               stdout = subprocess.DEVNULL,
                               check = True)
    
    done_time = time.time()
    
    stats = OrderedDict()
    stats[str(k) + 'mer_counting_time'] = done_time - start_time
    
    return(stats)


#given an input dsk hdf5 file, make an image with kmer counts
#mapping is the path to the table in parquet format that relates kmers to their positions in the image
#we do not need to save the ascii dsk kmer counts to disk, but the program requires a file
#so we will create a temporary file and delete it
def make_image(infile, outfolder, kmers, threads = 1, overwrite = False):
    Path(outfolder).mkdir(exist_ok = True)
    outfile = Path(infile).name.removesuffix(''.join(Path(infile).suffixes)) + '.png'    
    
    
    if not overwrite and (outfolder/outfile).is_file():
        eprint('File exists. Skipping image for file:',str(infile))
        return(OrderedDict())

    start_time = time.time()
    kmer_size = len(kmers.index[0])
    

    

    
    
    with tempfile.TemporaryDirectory(prefix='dsk') as outdir:
        # first, dump dsk results as ascii, save in a pandas df and merge with mapping
        # mapping has kmers and their reverse complements, so we need to aggregate
        # to get only canonical kmers
        for attempt in Retrying(stop = stop_after_attempt(5), wait = wait_random_exponential(multiplier=1, max=60)):
            with attempt:
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
        counts = kmers.join(counts).groupby(['x','y']).agg(sum).reset_index()
        counts.loc[:,'count'] = counts['count'].fillna(0)
        
        #Now we will place counts in an array, log the counts and rescale to use 8 bit integers
        array_width = kmers['x'].max()
        array_height = kmers['y'].max()

        
        #Now let's create the image:
        kmer_array = np.zeros(shape=[array_height, array_width])
        kmer_array[array_height-counts['y'],counts['x']-1] = counts['count'] + 1 #we do +1 so empty cells are different from zero-count 
        bins = np.quantile(kmer_array, np.arange(0,1,1/256))
        kmer_array = np.digitize(kmer_array, bins, right = False) - 1
        kmer_array = np.uint8(kmer_array)
        img = Image.fromarray(kmer_array, mode = 'L')
        
      
        #finally, save the image
        img.save(Path(outfolder)/outfile, optimize=True)
        
    done_time = time.time()
    stats = OrderedDict()
    stats['k' + str(kmer_size) + '_img_time'] = done_time - start_time
    
    return(stats)
    

###### Functions to train a cnn

#Function: select training and validation set given kmer size and amount of data (as a list).
#minbp_filter will only consider samples that have yielded at least that amount of data
def get_train_val_sets():
    file_path = [x.absolute() for x in (Path('images_' + str(kmer_size))).ls() if x.suffix == '.png']
    taxon = [x.name.split(label_sample_sep)[0] for x in file_path]
    sample = [x.name.split(label_sample_sep)[-1].split('_')[0] for x in file_path]
    n_bp = [int(x.name.split(sample_bp_sep)[-1].split('.')[0].replace('K','000')) for x in file_path]

    df = pd.DataFrame({'taxon': taxon,
                  'sample': sample,
                  'n_bp': n_bp,
                  'path': file_path
                 })
    
    if minbp_filter is not None:
        included_samples = df.loc[df['n_bp'] == 200000000]['sample'].drop_duplicates()
    else:
        included_samples = df['sample'].drop_duplicates()
        
    df = df.loc[df['sample'].isin(included_samples)]

    valid = (df[['taxon','sample']].
             drop_duplicates().
             groupby('taxon').
             apply(lambda x: x.sample(n_valid, replace=False)).
             reset_index(drop=True).
             assign(is_valid = True).
             merge(df, on = ['taxon','sample'])
        )

    train = (df.loc[~df['sample'].isin(valid['sample'])].
             assign(is_valid = False)
            )
    train = train.loc[train['n_bp'].isin(bp_training)]

    train_valid = pd.concat([valid, train]).reset_index(drop = True)
    return train_valid


## These functions use timm to create a fastai model

## Note: Functions using timm for choosing models have been adapted from this
## source: https://walkwithfastai.com/vision.external.timm
def create_timm_body(arch:str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("cut must be either integer or function")
    
def create_timm_model(arch:str, n_out, cut=None, pretrained=True, n_in=3, init=nn.init.kaiming_normal_, custom_head=None,
                     concat_pool=True, **kwargs):
    "Create custom architecture using `arch`, `n_in` and `n_out` from the `timm` library"
    body = create_timm_body(arch, pretrained, None, n_in)
    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children()))
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else: head = custom_head
    model = nn.Sequential(body, head)
    if init is not None: apply_init(model[1], init)
    return model

def timm_learner(dls, arch:str, loss_func=None, pretrained=True, cut=None, splitter=None,
                y_range=None, config=None, n_out=None, normalize=True, **kwargs):
    "Build a convnet style learner from `dls` and `arch` using the `timm` library"
    if config is None: config = {}
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = create_timm_model(arch, n_out, default_split, pretrained, y_range=y_range, **config)
    learn = Learner(dls, model, loss_func=loss_func, splitter=default_split, **kwargs)
    if pretrained: learn.freeze()
    return learn

#Function: create a learner and fit model, setting batch size according to number of training images
def train_cnn(df, 
              architecture,
              max_bs = 64,
              model_state_dict = None,
              epochs = 30, 
              freeze_epochs = 0,
              normalize = True,  
              callbacks = CutMix, 
              transforms = None,
              pretrained = False,
              loss_fn = LabelSmoothingCrossEntropyFlat()):
    
    
    #find a batch size that is a power of 2 and splits the dataset in about 10 batches
    batch_size = min(2**round(log(df[~df['is_valid']].shape[0]/10,2)), 64)
    batch_size = min(batch_size, max_bs)
    
    #start data block
    if 'is_valid' in df.columns:
        sptr = ColSplitter()
    else:
        sptr = RandomSplitter(valid_pct = args.validation_set_fraction)
        
    dbl = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter = sptr,
                       get_x = ColReader('path'),
                       get_y = ColReader('taxon'),
                       item_tfms = None,
                       batch_tfms = transforms
                      )
    
    #create data loaders with calculated batch size
    dls = dbl.dataloaders(df, bs = batch_size)
    

    #create learner
    learn = timm_learner(dls, 
                    architecture, 
                    metrics = accuracy, 
                    normalize = normalize,
                    pretrained = pretrained,
                    cbs = callbacks,
                    loss_func = loss_fn
                   ).to_fp16()
    
    #if there a pretrained model body weights, replace them
    #first, filter state dict to only keys that match and values that have the same size
    #this will allow incomptibilities (e. g. if training on a different number of classes)
    #solution inspired by: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/8
    if model_state_dict:
        old_state_dict = learn.state_dict()
        new_state_dict = {k:v for k,v in model_state_dict.items() 
                          if k in old_state_dict and
                             old_state_dict[k].size() == v.size()}
        learn.model.load_state_dict(new_state_dict, strict = False)
    
    #train
    learn.fine_tune(epochs = epochs, freeze_epochs = freeze_epochs, base_lr = 1e-3)
        
    
    return(learn)
    
