# tiny_reads fixture

`TestTaxon/sample1/sample1_R{1,2}.fastq.gz` is a tiny synthetic paired-end
fastq fixture, committed as static binary data so `image`-command tests can
run end-to-end without downloading real reads. It is generated once, not
regenerated per test run.

Generation recipe: 30 read pairs of 100bp each, random ACGT bases drawn from
`numpy.random.default_rng(seed=0)`, with an all-'I' quality string (Phred 40)
for every base. Reads are written as `@read{i}/1` / `@read{i}/2` records,
gzipped, and named `sample1_R1.fastq.gz` / `sample1_R2.fastq.gz` under a
`<taxon>/<sample>/` directory (`TestTaxon/sample1/`) matching the layout
`core/utils.py`'s folder-scan expects and the `_R1`/`_R2` pair-detection
regex in `commands/image.py`.

The exact bp counts asserted in `tests/test_image_e2e.py` follow
deterministically from this recipe, not from anything random:

- `4800` is the full-depth frame: 30 reads x (100bp - 2x10bp default
  `--trim-bp`) x 2 (paired) = 30 x 80 x 2.
- The `2000`/`1000` ladder below that comes entirely from `split_fastq`'s
  1-2-5 x 10^n stepping logic in `varKoder/commands/image.py` — pure integer
  arithmetic over the 4800bp total, not dependent on the random seed.
