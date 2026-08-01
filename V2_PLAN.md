# varKoder v2 — preparation plan (DRAFT for review)

**Status:** draft for maintainer review. Based on `release/1.8.0` (this branch's base).
No implementation has started — review and adjust before writing code. Delete or move to
an issue once superseded (as was done with the 1.8.0 design spec).

## Context

varKoder turns genomic k-mer frequencies into grayscale images ("varKodes"/CGR) and
classifies them with a CNN/ViT for DNA barcoding. v2 aims to: significantly improve speed,
modernize dependencies, shed non-pip/non-conda binaries, drop pickle model files, keep the
published NCBI SRA model working, and open the door to retraining the base model on
BioCLIP v2. This plan captures the design agreed after exploring the code and confirming
direction with the maintainer.

**Base branch:** all v2 work is based on **`release/1.8.0`** (soon to merge), not
`main`/1.7.1. 1.8.0 already did a large slice of the work; this plan builds *on* it.

### What 1.8.0 already delivers (do NOT rebuild)
- **safetensors + `config.json` model format** — `varKoder/core/model_io.py`
  (`save_varkoder_model`, `resolve_model`, `build_learner`). Config schema:
  `{architecture, label_names, is_multilabel, num_classes, input_size, normalize}`
  where `architecture` is the offline base timm name and `normalize` is `{mean,std}`.
- **Model resolution** across local dir / legacy `.pkl` / HF repo id (`resolve_model`).
- **Shared dataloader builder** `varKoder/core/preprocessing.py::make_dataloaders`
  (used by both train and inference) — currently a fastai `DataBlock`.
- **Custom archs relocated** to `varKoder/models/custom.py`.
- **pytest suite (81 tests)** + committed `tests/fixtures/tiny_reads/` + `tests/conftest.py`,
  plus `.github/workflows/test.yml` (single Python 3.11 job) and slow/network markers.
- **`push_to_hf.py`** publishes safetensors + config with an `allclose` **fidelity gate**.
- `DEFAULT_ARCHITECTURE = "vit_large_patch32_224"` (offline base name);
  `DEFAULT_MODEL = "brunoasm/vit_large_patch32_224.NCBI_SRA"`.
- Unparseable-filename crash fixed (`iter_varKoder_images(skip_unparseable=True)`).

**Crucial fact:** 1.8.0's safetensors weights *are the fastai `vision_learner` graph*
(timm body + fastai concat-pool/BN/dropout/2-linear head), and both train and query still
rebuild a fastai `Learner` around them. fastai is still a hard runtime dep, and `.pkl`
paths remain. v2's job is to **keep the config schema + safetensors format but replace the
fastai consumer with pure timm+torch, and delete the pkl paths.**

## Locked decisions (from the maintainer)
1. **Remove fastai entirely** from the runtime package → timm + pure PyTorch. Audit each
   fastai default when substituting (see "fastai defaults audit").
2. **Rust extension** (maturin/PyO3) replaces k-mer counting (`dsk`/`dsk2ascii`) and
   subsampling (`bbmap reformat.sh`); ships as wheels so `pip install` is self-sufficient.
3. **Keep `fastp`** (bioconda) for read cleaning.
4. **Hard break on `.pkl`** + ship a standalone converter that isolates fastai in an extra.
5. **Remove the custom archs** (`fiannaca2018`, `arias2022`) — peer-review only.
6. **Grayscale-native, backbone-aware channels**: varKodes are 8-bit grayscale; use 1-ch
   where the backbone allows, replicate-to-3ch only for RGB backbones (ViT/BioCLIP).
7. **Model head**: **new v2 models use a clean timm-native head**; a pure-torch
   **fastai-head compat loader** reads existing safetensors (NCBI SRA + user 1.8.0 models).
   `config.json` gains `head_type` (`"timm"` new; missing ⇒ `"fastai"`).
8. Maintain NCBI SRA compatibility; enable BioCLIP v2 as a future backbone.

---

## Workstream 1 — Remove fastai (model build, training, inference)

Keep the 1.8.0 safetensors/config format; swap the fastai consumer for pure timm+torch.

### 1a. Pure-torch model builders (`varKoder/models/build.py`, new)
- `build_timm_model(arch, num_classes, in_chans, pretrained) -> (model, spec)`: thin
  `timm.create_model(...)` wrapper (native head). Used for **new v2 training**
  (`head_type="timm"`). `spec` (a `BackboneSpec` dataclass) carries
  `input_size/mean/std/interpolation/crop_pct/fixed_input_size/classifier_name/in_chans`
  from `model.pretrained_cfg` — this is what makes BioCLIP-via-timm "just work".
- `build_fastai_compat_model(arch, num_classes, in_chans) -> model`: reimplements fastai's
  `vision_learner` graph in pure torch — timm body feature extractor + a `create_head`
  clone (`AdaptiveConcatPool2d` → `Flatten` → `BatchNorm1d` → `Dropout(0.25)` →
  `Linear(2·nf,512)` → `ReLU` → `BatchNorm1d` → `Dropout(0.5)` → `Linear(512,n_out)`),
  producing **byte-identical state_dict keys/shapes** to 1.8.0 so
  `load_state_dict(strict=True)` succeeds on existing safetensors. Used by the loader when
  `head_type=="fastai"`. Implementation must introspect the published NCBI SRA safetensors
  keys and match them exactly; gate with a strict-load + `allclose` fidelity test.
- `split_param_groups(model, spec)`: body vs `spec.classifier_name` (`'head'` for ViT,
  `'fc'` for ResNet) — for freeze/unfreeze and discriminative LRs.

### 1b. Pure-torch loader (rewrite `varKoder/core/model_io.py`)
- Keep `save_varkoder_model` / `resolve_model` shape; **add `head_type`** to config and a
  `preprocessing` block mirroring timm's `pretrained_cfg` keys.
- Replace `build_learner` (fastai) with `load_model(config, state_dict, device) ->
  LoadedModel{model, labels, is_multilabel, in_chans, preprocessing, threshold}`: builds
  `build_timm_model` or `build_fastai_compat_model` per `head_type`,
  `load_state_dict(strict=True)`, no fastai.
- Delete fastai imports (`Normalize`, `vision_learner`, `Learner`, `load_learner`) and the
  `_extract_normalize`/`_probe_input_size`/`_dummy_df` fastai machinery; normalization
  becomes plain values consumed by the transform.

### 1c. Pure-torch dataset/transforms (rewrite `core/imaging.py` + `core/preprocessing.py`)
- Replace `VarKodeImage(PILImage)` / `SelectRandomFrame(RandTransform)` with a
  `torch.utils.data.Dataset` (`VarKodeDataset`) + a plain transform. Port the multi-frame
  logic verbatim: training picks a random frame per `__getitem__` (open + `seek(idx)`,
  worker-safe); validation/query use frame 0; single-frame is a no-op. The fastai
  qualname-pickle coupling and the `import varKoder.core.imaging` side-effect in query.py
  both go away (no more pkl).
- Replace `make_dataloaders` (fastai `DataBlock`) with `build_transform(preproc, in_chans,
  is_training)` using `torchvision.transforms.v2`: grayscale load → channel handling →
  resize (honor `fixed_input_size`, Squish/BOX for the ViT) → `ToDtype(float32, scale)` →
  `Normalize(mean,std)`; training-only brightness/contrast (approx fastai lighting) +
  optional `RandomErasing`. Vocab ordering must reproduce fastai's
  `MultiCategoryBlock`/`CategoryBlock(sort=…)` (sorted unique; single-label pins order),
  persisted in config.

### 1d. Pure-torch training loop (rewrite `varKoder/commands/train.py`)
Move the engine to `varKoder/training/loop.py`; keep `TrainCommand` as orchestration.
Reuse the existing two-phase schedule and checkpoint/resume contract verbatim
(`run_fine_tune`, `CheckpointCallback` already write `last.pth` + `progress.json` +
`input_data.csv`). Replace fastai internals per the audit below. Metrics via `torchmetrics`
(`MultilabelAUROC`/`MultilabelPrecision`/`MultilabelRecall`, micro, excluding the
`low_quality:True` label). Keep the batch-size heuristic, `--no-metrics`, `--cpu`,
verbose→tqdm, and multi-GPU via `nn.DataParallel`.

### 1e. Pure-torch inference (rewrite `varKoder/commands/query.py::load_model`/`run`)
Replace `build_learner` + `learn.dls.test_dl` + `learn.get_preds` with a plain
`DataLoader(VarKodeDataset(...))` loop: multilabel → `sigmoid` + threshold; single-label →
`softmax` + argmax; `--include-probs` → `DataFrame(probs, columns=labels)`.
`_expand_query_items` and the output-table schema stay exactly as-is (locked by
`tests/test_query_command.py`).

### fastai defaults audit (the maintainer's explicit concern)
Each substitution documents and reproduces the hidden fastai default:

| fastai behavior | v2 replacement / value to preserve |
|---|---|
| `vision_learner` head (concat-pool 2·nf, BN, Dropout 0.25/0.5, 2 Linear, ReLU) | `build_fastai_compat_model` for load; new models use timm head |
| `freeze()` keeps **BatchNorm trainable** (`train_bn=True`) | replicate in freeze toggle |
| discriminative LR `slice(lo,hi)` across body/head groups | `OneCycleLR` `max_lr` list per param group |
| default optimizer = Adam w/ **decoupled wd** (`true_wd`, `wd=0.01`) | `torch.optim.AdamW(weight_decay=0.01)` |
| `fit_one_cycle` cycles **momentum** `(0.95,0.85,0.95)`; `div`, `div_final`, `pct_start` | `OneCycleLR(cycle_momentum=True, div_factor=div, ...)`; carry `pct_start=0.99` frozen / `0.3,div=5` unfrozen |
| `to_fp16` (GradScaler + autocast) | `torch.amp.autocast` + `torch.amp.GradScaler` |
| loss activation auto-applied in `get_preds` | explicit sigmoid (multilabel) / softmax (single) |
| `Normalize` from backbone stats (0.5/0.5 for ViT) | values from `config.normalize` / `pretrained_cfg` |
| `aug_transforms` lighting; MixUp/CutMix/RandomErasing α/prob | torchvision ColorJitter (approx) + loss-space MixUp/CutMix + `RandomErasing` |
| `PrecisionMulti/RecallMulti/RocAuc(average='micro')` excl. `low_quality:True` | torchmetrics micro metrics with label subset |
| `set_seed` pulls `fastai.torch_core.set_seed` | seed `random/numpy/torch` (+ optional cudnn.deterministic) directly |

**MixUp/CutMix subtlety:** timm's `Mixup` emits soft one-hot targets (wrong for the
multilabel `AsymmetricLossMultiLabel` path). Reproduce fastai's loss-space mixing
(`lam~Beta(α,α)`, permute batch, mix inputs, `lam·loss(y_a)+(1-lam)·loss(y_b)`) — works for
both single- and multi-label. Borrow only `cutmix_bbox_and_lam` from `timm.data.mixup`.

---

## Workstream 2 — Remove pkl back-compat + converter
- **`resolve_model`**: drop the `.pkl` file branch and the `from_pretrained_fastai`
  fallback; support only local safetensors dir + HF safetensors repo; raise a clear error
  for `.pkl` pointing at the converter. Remove `_PICKLE_WARNING` + fastai `load_learner`.
- **`export_trained_model`** (train.py): stop writing `trained_model.pkl` and the
  `DeprecationWarning`; write only `varkoder_model.safetensors` + `config.json` +
  `labels.txt` + `input_data.csv`.
- **Release gate:** the default NCBI SRA model must exist as safetensors+config on HF
  (1.8.0's `push_to_hf` produces this) — confirm uploaded before v2 default `query` works.
- **Converter** `varKoder/tools/convert_pkl_model.py` → console script
  `varkoder-convert-model old.pkl out_dir/`, guarded by `pip install "varKoder[convert]"`
  (fastai imported lazily). Extracts vocab, multilabel flag, base arch, writes safetensors
  with `head_type="fastai"`. For **users' own** `.pkl`; the default NCBI SRA is handled by
  re-publishing, not per-user conversion.

---

## Workstream 3 — Rust extension + de-subprocess the pipeline
All shell-outs live in `varKoder/commands/image.py` (the 1614-line "god module").

### 3a. Rust crate `varKoder._native` (maturin/PyO3)
- `subsample_and_count(path, k, target_bps, seed, threads) -> [(codes, counts, actual_bp)]`
  fuses subsampling + canonical k-mer counting (replaces `reformat.sh` + `dsk` +
  `dsk2ascii` + the pandas ASCII round-trip). Deterministic, thread-count-independent read
  selection; rayon only parallelizes counting.
- `write_subsamples(...)` for `image --no-image`; `scan_fastq(path)->(reads,bp)` replacing
  `gunzip|wc`; `count_canonical_kmers(path,k,threads)` for the parity harness.
- Crates: `needletail`, `flate2`/`miniz_oxide` (portable, no system zlib), `rayon`,
  `rand`/`rand_chacha`, `pyo3`+`numpy`+`ndarray`. `abi3-py311`.
- **Parity-preserving scope for v2.0:** keep the pixel-array assembly + 256-quantile
  log-binning **in numpy** (`compute_kmer_array` unchanged except sourcing counts from Rust
  by integer code). Bit-identical images given identical counts. Fusing pixel assembly into
  Rust is deferred to v2.1.

### 3b. Pipeline extraction into `varKoder/processing/` (empty pkg today)
Split `image.py` into `reads.py` (fastp cleaning; keep `MALLOC_ARENA_MAX=1`), `sampling.py`
(target-bp math in Python + Rust calls), `kmers.py` (Rust bridge + `build_pixel_index`),
`imageio.py` (`compute_kmer_array`/`make_image`/`make_multiframe_image` verbatim),
`pipeline.py` (`run_clean2img`). Repoint `query.py` imports from `commands.image` →
`processing.*`. Preserve all filename + PNG tEXt conventions and stats keys exactly.

### 3c. Light shell-outs → pure Python
Replace `gunzip|wc` (image.py:127) with `scan_fastq`; replace `cat|pigz`
(image.py:652-663) with a Python `gzip`/`shutil.copyfileobj` writer, pigz **optional** via
`shutil.which(PIGZ_CMD)` and **fixing the hardcoded `"pigz"`** at line 658.

### 3d. Concurrency
`multiprocessing.Pool` gets an `initializer` that builds `kmer_mapping` once per worker
(stop pickling the DataFrame into every task); bound rayon to `cpus_per_thread`; drop the
`tenacity` retries. v2.1: fused Rust pixel path + `ThreadPoolExecutor` over samples.

### 3e. Parity + behavior change
- Gate: Rust `count_canonical_kmers` == `dsk`+`dsk2ascii` for k=5–9 (incl. reads with `N`);
  bit-identical PNG arrays with subsampling bypassed, for both `varKode` and `cgr`.
- **Documented behavior change:** v2 subsampling won't reproduce bbmap's exact read
  selection, so v2 images differ from v1 for the same input (both valid random samples).
  Determinism preserved for a fixed `--seed`.

---

## Workstream 4 — Remove custom architectures
Delete `varKoder/models/custom.py`; remove `CUSTOM_ARCHS` (`config.py`) and every reference
in `model_io.py` (`recover_architecture`, `build_learner` branch), `preprocessing.py`, and
`train.py` (`build_custom_model`, the `Learner` branch). Update the `--architecture` help.
Delete `tests/test_custom_models.py` and the custom-arch training smoke test; keep a
timm-arch smoke test.

---

## Workstream 5 — Packaging / deps / build / CLI / docs
- **`pyproject.toml` → maturin backend**, bump **2.0.0**. Core deps (modern pins): `torch`,
  `timm>=1.0.15`, `safetensors`, `torchvision`, `torchmetrics`, `huggingface_hub` (no
  `[fastai]`), `pillow`, `pandas`, `pyarrow`, `numpy`, `humanfriendly`, `tenacity`, `tqdm`.
  **Remove** `fastai` (+ `accelerate`). Extras: `convert = ["fastai","huggingface_hub[fastai]"]`,
  `bioclip = ["open_clip_torch"]`, `dev = ["pytest","ruff","py-spy","maturin"]`. Add
  `varkoder-convert-model` script, `[tool.maturin]`, and a root `Cargo.toml`. Drop
  `MANIFEST.in`.
- **conda envs**: drop `dsk` + `bbmap`; keep `fastp`, `pigz`, `sra-tools` (tests only);
  stop pinning the ML stack. Removes the macOS manual-dsk step.
- **Dockerfile**: multi-stage — builder compiles the Rust wheel; runtime keeps the CUDA
  torch base + `fastp`/`pigz`, `pip install --no-deps` the wheel. Drop dsk-from-source,
  bbmap, fastai/accelerate/`toml`.
- **CLI** (`cli.py`): modularize `setup_parser` with a shared `preprocess_parent` for the
  flags duplicated between `image`/`query` (keep the exact flag surface; `-m` differs by
  command). Update pkl/fastai-doc help strings. Optional `--in-chans {auto,1,3}` on `train`.
- **Docs**: install (drop dsk/bbmap, note bundled Rust + prebuilt wheels), model format
  (safetensors+config, `head_type`), security note.

---

## Workstream 6 — Testing / CI (extend 1.8.0's suite)
- **Update the fastai-based fixtures** in `tests/conftest.py` (`tiny_timm_learner`,
  `synthetic_images`) to the pure-torch builders.
- **Add**: Rust parity tests; sequence-processing unit tests (`clean_reads`, sampling);
  pure-torch model save/load round-trip; **NCBI SRA fidelity test** (network-marked)
  asserting v2 predictions match the 1.8.0 baseline within tolerance.
- **CI**: extend `test.yml` with `ruff`, a Python 3.11–3.13 matrix, and a Rust toolchain +
  `maturin develop`; add `wheels.yml` (`PyO3/maturin-action`, linux x86_64/aarch64 + macOS
  arm64/x86_64, `abi3`, PyPI Trusted Publishing on release); wire the image e2e test into a
  job that installs `fastp`/`pigz` via micromamba.

---

## Backbone-aware grayscale (spans WS1)
Load frames as grayscale `L`. Drive channels from `in_chans` (stored in config): `3` →
replicate to 3×H×W (preserves the NCBI SRA ViT exactly); `1` → single channel, built via
`timm.create_model(arch, in_chans=1)` (timm sums pretrained RGB conv weights into 1-ch).
Default: `1` for from-scratch / timm-pretrained backbones; **force `3` when loading or
warm-starting a 3-ch model** (NCBI SRA). The stored value prevents a train/query mismatch.

---

## Critical gates & risks
1. **NCBI SRA loads without fastai and reproduces predictions.** The compat loader must
   `load_state_dict(strict=True)` the published NCBI SRA safetensors and match 1.8.0's
   `build_learner` output within `allclose` on fixed images. Gate before deleting pkl paths.
2. **Head-graph reproduction** — introspect the published safetensors keys to match fastai's
   exact head module names/shapes.
3. **fastai training-default parity** — the audit table; expect close-not-identical training
   curves (OneCycle/optimizer defaults differ); validate on the tiny fixture.
4. **Rust count parity** + the documented subsampling behavior change.
5. **Grayscale `in_chans` gating** — never rebuild 3-ch weights at `in_chans=1`.

## Recommended sequencing
- **Phase 0 (done):** branch reset onto `release/1.8.0`.
- **Phase 1 (independent, lower risk):** remove custom archs (WS4); Rust scaffold + parity
  harness + pipeline extraction into `processing/` + light shell-out removal (WS3a–3e);
  packaging → maturin + wheels CI + docs (WS5, WS6 CI). No model-behavior change yet.
- **Phase 2 (fastai removal):** pure-torch model builders + loader (WS1a–1b) gated by the
  NCBI SRA fidelity test; pure-torch dataset/transforms (WS1c); training loop + metrics
  (WS1d); query inference (WS1e); drop pkl paths + converter (WS2); update fixtures (WS6).
- **Phase 3 (v2.1):** fused Rust pixel assembly + thread concurrency; BioCLIP v2 backbone
  path (`open_clip` optional extra + a `create_backbone` branch).

## Verification
- Unit + Rust-parity + model round-trip: `pytest -m "not slow and not network"`.
- NCBI SRA fidelity: `pytest -m network` (loads the HF safetensors, compares to a saved
  baseline / the 1.8.0 loader within tolerance).
- End-to-end on `tests/fixtures/tiny_reads/`: `image` → `train` (1 epoch, CPU, timm arch)
  → `query`; assert safetensors+config+labels+input_data outputs and `predictions.csv`
  columns. The ~25-min `tests/03` SRA run remains the full integration check.
- Profiling before/after with `/usr/bin/time -v` + `py-spy`/`scalene` on the Bembidion
  workload, capturing wall/CPU/RSS/subprocess-count/per-stage `stats.csv` timings.

## Key files
- Reuse/extend: `varKoder/core/model_io.py`, `varKoder/core/preprocessing.py`,
  `varKoder/core/imaging.py`, `varKoder/commands/{train,query,image}.py`,
  `varKoder/core/{config,utils}.py`, `varKoder/cli.py`, `tests/conftest.py`,
  `xtra_scripts/push_to_hf.py`, `pyproject.toml`, `Dockerfile`, `.github/workflows/`.
- New: `varKoder/models/build.py`, `varKoder/training/loop.py`,
  `varKoder/tools/convert_pkl_model.py`,
  `varKoder/processing/{reads,sampling,kmers,imageio,pipeline}.py`, root `Cargo.toml` +
  `src/` Rust crate, `.github/workflows/wheels.yml`.
- Delete: `varKoder/models/custom.py`, `varKoder/models/metrics.py` (dead), `.pkl` paths.
