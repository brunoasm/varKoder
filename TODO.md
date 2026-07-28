# TODO before releasing v1.8.0

Two pieces of work agreed after reconciling the safetensors branch with `main`.
Delete this file (or move anything unfinished into an issue) before tagging.

---

## 1. Rethink the pytest suite and broaden coverage

### Where we are

32 tests, and 26 of them sit in just two areas — the suite was written to guard
the safetensors work, not the program as a whole.

| Area | Tests | File |
| --- | --- | --- |
| Model save / resolve / rebuild / fidelity | 16 | `test_model_io_{save,roundtrip,resolve,compat}.py`, `test_push_fidelity.py` |
| Preprocessing + multi-frame pipeline | 9 | `test_preprocessing.py` |
| Custom architectures | 3 | `test_custom_models.py` |
| `export_trained_model` | 1 | `test_train_export.py` |
| `QueryCommand.load_model` | 1 | `test_query_load.py` |
| `train` CLI defaults | 1 | `test_cli_defaults.py` |
| Fixture smoke test | 1 | `test_smoke.py` |

The whole suite runs in ~7 s. That speed is worth protecting.

### What is not covered at all

- **`core/utils.py`** — nothing. Includes `get_metadata_from_img_filename`,
  which is what crashes in item 2 below, plus `iter_varKoder_images`,
  `get_varKoder_frame_sizes`, `parse_bp_human_readable`,
  `format_bp_human_readable`.
- **`image` command** — k-mer counting, image array shape per mapping/k, APNG
  writing under `--stack`, tEXt metadata, quality flagging.
- **`convert` command** — nothing, single-frame or multi-frame.
- **The training loop** — `export_trained_model` is tested, but `fit`/`fine_tune`,
  per-epoch checkpointing and `--resume` are not.
- **Sequence processing** — fastq/fasta handling, read cleaning, dsk, dedup.
- **`query` beyond `load_model`** — `_expand_query_items`, `--all-frames` frame
  extraction, `predictions.csv` contents, threshold handling, temp-dir cleanup.
- **Network paths** — the HF `from_pretrained_fastai` fallback and
  `hf_hub_download`. Deliberate (offline default), but see markers below.

Consequence: the fast suite would not catch a regression in image generation,
conversion, or the query output table. Only `tests/03` would, and that takes
~25 minutes and needs the SRA download.

### Proposed work, highest value first

1. **`core/utils.py` unit tests.** Cheapest coverage in the repo, pure functions,
   no fixtures needed. Round-trip `format_bp_human_readable` /
   `parse_bp_human_readable`; parse valid varKode names across the naming
   generations the function already claims to support (current
   `sample@bp+mapping+kN.png`, the v0.x two-field form, and multi-frame
   `sample@stack+mapping+kN.apng`); `iter_varKoder_images` finds `.png` and
   `.apng` and ignores everything else (`config.IMAGE_GLOBS`, config.py:26);
   `get_varKoder_frame_sizes` on valid, absent and malformed metadata — `main`
   already made that tolerant in e294531, so lock it.

2. **`query` output-table tests.** The user-facing contract, currently
   integration-only. Assert the exact column set for single-label vs multi-label
   (they differ, and `best_pred_prob` is single-label only); `_expand_query_items`
   yields one item per file by default and one per frame under `--all-frames`,
   with `report_path` staying the original file and per-frame `query_basepairs`;
   `--threshold` boundary behaviour. Also lock the temp-dir cleanup that `main`
   fixed in cc4a9c8 and d6be8ad (`--int-folder`, `--keep-images`) — those are
   exactly the kind of leak that silently regresses.

3. **`convert` tests.** Reuse the existing `multiframe_images` fixture in
   `conftest.py`. Single-frame remap produces the expected size and preserves
   metadata; a stack remaps every frame and comes back with the same frame count
   and order (frame 0 still representative); output naming is
   `sample@stack+mapping+kN.apng`.

4. **A tiny end-to-end `image` test.** One small synthetic fastq (or fasta) →
   assert an image is produced with the expected dimensions for a given
   mapping/k, the filename parses back to the inputs, and `--stack` yields one
   APNG whose frames are ordered largest-bp-first with `varkoderFrameSizes` and
   `varkoderFormatVersion` set. This is the biggest coverage win but the most
   work; it needs a committed miniature fastq fixture, which the repo does not
   have today (`tests/Bembidion` is downloaded, not committed).

5. **A 1-epoch training smoke test.** Custom arch on CPU over the synthetic
   fixture: assert the four output files appear, then resume from the checkpoint
   and assert `progress.json` advances. Guards the resume feature, which has no
   automated coverage at all.

### Suite-level decisions to make

- **Markers.** Introduce `@pytest.mark.slow` (and maybe `network`) so the default
  `pytest` run stays a few seconds while heavier tests are opt-in
  (`-m "not slow"` by default via `addopts` in `pyproject.toml`). Needed before
  adding items 4 and 5, and would let us finally test the HF fallback under
  `-m network`.
- **CI.** `.github/workflows/` has only `docker.yml`; nothing runs pytest. Add a
  workflow — decide the Python/OS matrix (package requires `>=3.11`; we develop
  on ARM macOS, users are mostly Linux).
- **README.** The testing section documents only `01_download_fastqs.sh` and
  `03_test_installation.sh`. Mention the pytest suite and that it needs no
  downloads.
- **Fixture strategy.** Decide whether to commit a miniature fastq/fasta so
  image-level tests can run without SRA. Keep it small enough to live in git.

---

## 2. Fix the unparseable-filename crash

### The bug

One file whose name does not parse aborts an entire run:

```
ValueError: invalid literal for int() with base 10: '7 2'
```

Raised at `varKoder/core/utils.py:306`, `int(img_kmer_size[1:])`, inside
`get_metadata_from_img_filename`. It is called while scanning input directories
from all three commands:

- `varKoder/commands/train.py:400` (`collect_images`)
- `varKoder/commands/query.py:237` (`_expand_query_items`)
- `varKoder/commands/convert.py:181`

So a single stray file kills a training run that may be hours in.

### How to reproduce

```bash
touch "tests/images/whatever@00500K+cgr+k7 2.png"
varKoder train --overwrite tests/images /tmp/out
```

Real-world trigger: this repo lives under an iCloud-synced `~/Documents`, and
deleting then regenerating `tests/images*` makes iCloud restore the
not-yet-synced deletions as `...+k7 2.png` conflict copies. Dropbox and OneDrive
produce the same class of name. It cost two bogus integration runs on
2026-07-28. Users keeping varKodes in a synced folder will hit it.

### Decisions needed

- **Skip or fail?** Recommend **skip with a warning** plus a count in the summary
  ("ignored N files whose names are not varKode names"), because a recursive scan
  of a user's directory will legitimately meet foreign files. A hard failure is
  defensible for `train` only if the message names the offending files — the
  current opaque `ValueError` is the worst of both.
- **Where to enforce it.** Cleanest is probably to have `iter_varKoder_images`
  (utils.py:149) yield only names that parse, so all three commands inherit the
  behaviour, rather than patching three call sites. Check that `convert.py:181`
  and `query.py:237` actually route through it first — `train.py:399` does.
- **Which release.** `core/utils.py` is untouched by the safetensors branch
  (zero diff vs `main`), so this affects **1.7.1 and earlier too**. Decide: fix
  in 1.8.0, or also cut a 1.7.2 patch.
- **Scope of tolerance.** Note `utils.py:297` does
  `name.removesuffix('.png')`, so double-check `.apng` and unexpected extensions
  behave sensibly once names are validated.

### Tests to add with the fix

Pure-function tests in the new `tests/test_utils.py` from item 1: a set of
malformed names (`+k7 2`, missing `@`, missing `+`, empty k-mer field,
non-numeric bp, a plain `notavarkode.png`) are rejected/skipped rather than
raising; and a directory containing one good image plus each malformed name
yields exactly the good one. Then an integration-level assertion that `train`
completes with a warning instead of dying.
