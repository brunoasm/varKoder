#!/bin/sh


tail -n +2 Bembidion.csv | while read line
do
echo working on $line
accession=$(echo $line | cut -d , -f 1)
taxid=$(echo $line | cut -d , -f 2) 

fastq-dump -N 10000 -X 510000 --skip-technical --gzip --read-filter pass --readids --split-spot --split-files --outdir ./Bembidion/$taxid/$accession $accession
done

echo DONE

exit
