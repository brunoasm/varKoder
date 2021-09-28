#!/usr/bin/env python 

#imports used for all commands
from pathlib import Path
from collections import OrderedDict, defaultdict
from io import StringIO
from PIL import Image
from matplotlib import cm
from tenacity import Retrying, stop_after_attempt, wait_random_exponential
import pandas as pd, numpy as np, tempfile, shutil, subprocess, functools
import re, sys, gzip, time, humanfriendly, random, multiprocessing, math

from scipy.stats import boxcox
from scipy.stats import rankdata


from fastai.torch_core import set_seed


# defining a function to print to stderr more easily
# idea from https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    

# this function process the input file or folder and returns a table with files
def process_input(inpath):
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
    
    #let's check if destination files exist, and skip them if we can
    
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
                                ##### CAUSING BUG, MISMATCH IN R1 AND R2
                                #if ((line_n % 4 == 0) and 
                                #     (write_out is True) and 
                                #     (max_bp is not None) and 
                                #     (initial_bp > (3 * max_bp))):
                                #    retained_bp = initial_bp
                                #    write_out = False
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
def split_fastq(infile, 
                outprefix, 
                outfolder, 
                min_bp = 50000, 
                max_bp = None, 
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
    elif int(nsites) > min_bp:
        sites_per_file = [min(int(nsites), int(max_bp))]
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
        
    # now we will use bbtools reformat.sh to subsample
    outfs = [Path(outfolder)/(outprefix + '_' + str(int(bp/1000)).rjust(8, '0') + 'K' + '.fq.gz') for bp in sites_per_file]
    
    if all([f.is_file() for f in outfs]):
        if not overwrite:
            eprint('Files exist. Skipping subsampling for file:',str(infile))
            return(OrderedDict())
    
    for i, bp in enumerate(sites_per_file):
        outfile = Path(outfolder)/(outprefix + '_' + str(int(bp/1000)).rjust(8, '0') + 'K' + '.fq.gz')
        
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
                                '-nb-cores', str(1),
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
                                                   '-nb-cores', str(1),
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
        #kmer_array = np.zeros(shape=[array_height, array_width])
        #kmer_array[array_height-counts['y'],counts['x']-1] = counts['count']
        #kmer_array = (kmer_array - np.min(kmer_array))/np.ptp(kmer_array) + 1 #data transformation will require positive numbers, so we add 0.1
        #kmer_array = boxcox(counts['count'], -0.3)
        #kmer_array = (kmer_array - np.min(kmer_array))/np.ptp(kmer_array)
        
        #kmer_array = np.zeros(shape=[array_height, array_width])
        #kmer_array[array_height-counts['y'],counts['x']-1] = counts['count']
        #kmer_array = rankdata(kmer_array, method = 'dense').reshape(kmer_array.shape)
        #kmer_array = np.uint8(kmer_array/np.max(kmer_array) * 255)
        
        kmer_array = np.zeros(shape=[array_height, array_width])
        kmer_array[array_height-counts['y'],counts['x']-1] = counts['count']
        bins = np.quantile(kmer_array, np.arange(0,1,1/256))
        kmer_array = np.digitize(kmer_array, bins, right = False) - 1
        kmer_array = np.uint8(kmer_array)

        #kmer_array = np.uint8(255*(kmer_array - np.nanmin(kmer_array))/ (np.nanmax(kmer_array) - np.nanmin(kmer_array)))
        img = Image.fromarray(kmer_array, mode = 'L')
        
        #mapped_img = np.uint8(255*cm.turbo(kmer_array))
        #img = Image.fromarray(mapped_img)
      
        #finally, save the image
        
        img.save(Path(outfolder)/outfile, optimize=True)
        
    done_time = time.time()
    stats = OrderedDict()
    stats['k' + str(kmer_size) + '_img_time'] = done_time - start_time
    
    return(stats)
    

##train a cnn to recognize kmer images. input_table should have a column with species ID and another with associated image paths
##kwargs are optional keyword arguments passed to fine_tune()
##the final model will be saved to outfolder, as well as a text file listing the names of species in the order that they correspond to classes in the model
def train_cnn(input_table, outfolder, frozen_epochs, epochs):
    pass
    
