# TODO before releasing v1.8.0

Two pieces of work agreed after reconciling the safetensors branch with `main`.
Delete this file (or move anything unfinished into an issue) before tagging.

---

## 1. Rethink the pytest suite and broaden coverage

### Where we are

66 tests. The two biggest areas are now `core/utils.py` (added by item 1
below) and model save/resolve/rebuild/fidelity — the suite was originally
written to guard the safetensors work, not the program as a whole, and that's
still true outside those two areas.

| Area | Tests | File |
| --- | --- | --- |
| `core/utils.py` (parsing, formatting, image discovery) | 33 | `test_utils.py` |
| Model save / resolve / rebuild / fidelity | 16 | `test_model_io_{save,roundtrip,resolve,compat}.py`, `test_push_fidelity.py` |
| Preprocessing + multi-frame pipeline | 9 | `test_preprocessing.py` |
| Custom architectures | 3 | `test_custom_models.py` |
| `export_trained_model` | 1 | `test_train_export.py` |
| `QueryCommand.load_model` | 1 | `test_query_load.py` |
| `train` CLI defaults | 1 | `test_cli_defaults.py` |
| `convert` arbitrary-name regression (item 2 fix) | 1 | `test_convert_arbitrary_names.py` |
| Fixture smoke test | 1 | `test_smoke.py` |

The whole suite runs in a few seconds. That speed is worth protecting.

### What is not covered at all

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

1. **DONE** — **`core/utils.py` unit tests.** Cheapest coverage in the repo,
   pure functions, no fixtures needed. Landed as `tests/test_utils.py` (33
   tests): round-trips `format_bp_human_readable` / `parse_bp_human_readable`;
   parses valid varKode names across the naming generations the function
   already claims to support (current `sample@bp+mapping+kN.png`, the v0.x
   two-field form, and multi-frame `sample@stack+mapping+kN.apng`);
   `iter_varKoder_images` finds `.png` and `.apng`, ignores everything else,
   recurses into subdirectories, and skips unparseable names by default (see
   item 2 below); `get_varKoder_frame_sizes` on valid, absent and malformed
   metadata.

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

## 2. Fix the unparseable-filename crash — RESOLVED

A single foreign or sync-conflict filename (e.g. an iCloud/Dropbox/OneDrive
`...+k7 2.png` conflict copy) used to raise an opaque `ValueError` out of
`get_metadata_from_img_filename` and abort an entire `train`/`query` run.

Fix: `iter_varKoder_images` in `varKoder/core/utils.py` gained a
`skip_unparseable` parameter, defaulting to `True`. In that default mode it
skips files whose names don't parse instead of raising, printing one warning
with the total count plus a listing of the skipped paths (capped to the first
10, with a "... and N more" line beyond that). All three commands (`train`,
`query`, `convert`) route their directory scans through this function, so they
inherit the behaviour automatically — except `convert`, which was given an
explicit `skip_unparseable=False` override, since it has a real, working
feature of remapping arbitrarily-named images when the caller passes explicit
`--input-mapping`/`--kmer-size` overrides; an unconditional filter would have
silently broken that.

Per the user's own decision at the time, this fix ships in 1.8.0 only — it is
not backported to a 1.7.2 patch, even though `core/utils.py` was otherwise
untouched by the safetensors branch and the bug also affects 1.7.1 and
earlier.

Tests: `tests/test_utils.py` covers malformed single-frame and multi-frame
names (rejected by `get_metadata_from_img_filename`, skipped-with-warning by
`iter_varKoder_images`), the warning-cap behaviour, and that `skip_unparseable`
can be disabled; `tests/test_convert_arbitrary_names.py` locks in `convert`'s
arbitrary-name override; and a `TrainCommand.collect_images` test in
`tests/test_utils.py` confirms `train` completes with a warning instead of
raising.
