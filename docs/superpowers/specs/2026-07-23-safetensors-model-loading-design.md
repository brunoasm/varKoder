# Weights-only (safetensors) model loading

**Date:** 2026-07-23
**Status:** Design — awaiting review

## Problem

varKoder distributes and loads its neural network as a pickled fastai `Learner`
(`.pkl`). The pre-trained model on Hugging Face is loaded via
`from_pretrained_fastai`, which downloads and **unpickles** `model.pkl`. Python
pickle can execute arbitrary code on load, so anyone querying with the default
model is trusting a remote pickle (Hugging Face itself flags this repo's
`model.pkl` as unsafe: 89 imports). Locally trained models and the fine-tuning
input path have the same exposure.

We want the security posture of loading **weights only** (safetensors) instead of
executable pickles, across the whole pipeline, while keeping predictions
identical to the current model and preserving backward compatibility during a
transition period.

## Goals

- `query` loads the model from safetensors weights + a small JSON config, for
  both the Hugging Face default model and locally trained models.
- `train` fine-tunes from safetensors weights, and exports safetensors.
- Predictions from the weights-only path are **identical** to the current
  pickled model (this is a hard requirement, verified before publishing).
- Backward compatible during the transition: legacy `.pkl` still loads (with a
  security warning), the HF repo keeps `model.pkl`, and `train` keeps writing a
  `.pkl` (with a deprecation warning) in addition to safetensors.

## Non-goals

- Removing legacy files from the HF repo (the maintainer does this later, for
  newer model versions).
- Changing the model architecture, image generation, or k-mer logic.
- Rewriting inference in pure timm/torch — we keep fastai and only replace the
  serialization format.

## Key findings that shape the design

1. **fastai head ≠ timm head.** The `model.safetensors` currently on HF was
   produced by `xtra_scripts/push_to_hf.py`, which builds a *fresh timm model
   with a timm classifier head* and copies weights across by name+shape match.
   fastai's `vision_learner` uses a different head (concat-pool → BatchNorm →
   Linear) whose final layer shape differs from timm's `head`. The head weights
   therefore never transfer — that file carries a randomly-initialized head. It
   is **not usable for faithful inference**.

2. **The timm `model.safetensors` has a real job anyway.** Old varKoder's
   default `train` path uses `--architecture hf-hub:brunoasm/…` with
   `pretrained=True`, which makes timm download that file as the **pretrained
   body**; `vision_learner` discards and recreates the head, so the random head
   never mattered there. We must not break that file for old installs during the
   transition.

3. **The faithful weights are the fastai `Learner.model.state_dict()`.** Saving
   that state dict (body `0.model.*` + trained head `1.*`) to safetensors and
   loading it back into an identically-constructed `vision_learner` reproduces
   the pkl exactly, because it is the same module built the same way. The same
   file also serves fine-tuning: loaded with `strict=False`, the body transfers
   and the head is skipped on shape mismatch — the current
   fresh-head-on-trained-body behavior.

## Decisions (from brainstorming)

- **Scope:** whole pipeline (query, train export, fine-tuning input).
- **Load path:** rebuild the fastai `Learner` from safetensors (keep fastai;
  remove pickle). Not a pure-timm rewrite.
- **Single new artifact** for new code: `varkoder_model.safetensors` +
  `config.json`. varKoder no longer relies on timm's automatic `hf-hub:`
  download for the default body.
- **Default training body:** `--pretrained-model` defaults to `DEFAULT_MODEL`
  (the HF repo); `--architecture` drops to the base name for the
  scratch/in21k case.
- **Backward compatibility:** legacy `.pkl` still loads with a security warning;
  the HF repo keeps `model.pkl` (and the legacy timm files); `train` writes both
  `.pkl` (with a deprecation warning) and safetensors.

## Artifact format

A varKoder model is **two files**, used identically for a local directory or an
HF repo:

- **`varkoder_model.safetensors`** — the fastai `Learner.model.state_dict()`,
  saved with `safetensors`. Stored in **fp32** with contiguous tensors (training
  may run in fp16 on GPU; export casts to fp32 for portability). Keys are the
  fastai module structure: body under `0.model.*`, head under `1.*`.
- **`config.json`**:
  ```json
  {
    "architecture": "vit_large_patch32_224",
    "label_names": ["...", "..."],
    "is_multilabel": true,
    "num_classes": 28643
  }
  ```
  - `architecture` is the **base timm name**, never the `hf-hub:` form (so
    `create_model(architecture, pretrained=False)` and rebuilding work offline).
  - `label_names` is the vocab in exact index order (index *i* ↔ head output *i*).
  - `is_multilabel` replaces the current `"MultiLabel" in str(learn.loss_func)`
    sniff, which is unavailable without the pickle.
  - Image preprocessing is **not** stored; it is reconstructed from
    `architecture` exactly as `train_nn` does, to avoid a second source of truth
    that could drift. For **timm** archs this means re-deriving resize + mean/std
    from `create_model(architecture).default_cfg`; for **custom** archs
    (`fiannaca2018`, `arias2022`) it means no resize and no normalization, per the
    custom training branch.

### HF repo, transition state

Keep (untouched, for old installs): `model.pkl` (old `query`), the timm
`model.safetensors` and `pytorch_model.bin` (old default `train` body).
**Add:** `varkoder_model.safetensors`. **Update:** `config.json` to add
`is_multilabel` (timm ignores the extra key, so old training is unaffected;
`architecture` is already the base name).

### HF repo, end state (maintainer, later)

`varkoder_model.safetensors` + `config.json` only.

## Components

### New module: `varKoder/core/model_io.py`

Single source of truth for reading/writing the format so query, train, export,
and the push script agree.

- `save_varkoder_model(learn, outdir, *, architecture, is_multilabel)` — write
  `varkoder_model.safetensors` (fp32, contiguous state dict) + `config.json`.
  `architecture` and `is_multilabel` are passed **explicitly** by the caller
  (train knows both directly; the push script derives them from the pkl). A
  helper `recover_architecture(learn)` reads
  `learn.model[0].model.default_cfg["architecture"]` for timm archs, or returns
  the custom-arch name, for callers that only have a learner.
- `resolve_model(source)` → `(state_dict, config)`. `source` may be:
  - a **local directory** containing the two files → read directly;
  - a **local `.pkl`** (legacy) → emit a **security warning**, `load_learner`,
    extract `model.state_dict()`, and synthesize `config` (architecture via
    `recover_architecture`, `is_multilabel` from the loss function);
  - an **HF repo id** → `hf_hub_download` the two files, then read. If the repo
    lacks the safetensors/config (an old-style repo), **fall back** to the legacy
    `from_pretrained_fastai` (download `model.pkl`) behind the security warning,
    and synthesize `config` from the loaded learner. This preserves querying of
    old-style HF repos.
- `build_learner(config, dls_or_vocab, device)` — rebuild the learner with the
  vocab fixed to `config["label_names"]`, reusing a transform/DataBlock builder
  refactored out of `train_nn` so inference preprocessing is byte-identical to
  training. Two branches, mirroring `train_nn`:
  - **timm architecture:** `vision_learner(pretrained=False)`; preprocessing
    (resize + normalize) re-derived from `create_model(architecture).default_cfg`.
  - **custom architecture** (`architecture in CUSTOM_ARCHS`): map the name to the
    class, instantiate `Model(num_classes=len(label_names), is_multilabel=…)`,
    wrap in a bare `Learner`; **no resize, no normalization** (images stay in
    `[0,1]` at native size), matching the custom training branch. The current
    custom models (`fiannaca2018`, `arias2022`) have no lazy layers, so parameter
    shapes are fixed by `num_classes` and no input-size metadata is needed. (A
    future custom arch using `LazyLinear` would require storing input size in
    `config.json`.)

### Refactor in `train_nn`

Extract the item-transform + batch-transform + DataBlock construction (currently
inline in `train.py:329-385`) into a shared helper used by both training and
`build_learner`, parameterized by `architecture`, `is_multilabel`, and vocab.
This is the mechanism that guarantees inference preprocessing matches training.

Building an inference `dls` without training data: construct a minimal
`DataLoaders` with the vocab pinned explicitly (`CategoryBlock(vocab=…)` /
`MultiCategoryBlock(vocab=…)`) so head-output order matches `label_names`, then
`learn.dls.test_dl(real_images)` for prediction as query already does.

### `query.load_model()` (rewrite)

`resolve_model(args.model)` → `build_learner` → `load_state_dict(strict=True)` →
existing `get_preds`/`test_dl` prediction code. Multi-label vs single-label is
read from `config["is_multilabel"]` rather than the loss function. Legacy `.pkl`
still works via the old `load_learner` / `from_pretrained_fastai` path behind the
security warning.

### `train` (changes)

- **Export:** in addition to `labels.txt` / `input_data.csv`, write **both**
  `trained_model.pkl` (via `learn.export`, emitting a **`DeprecationWarning`** +
  `eprint` that pkl export is deprecated and will be removed in a future release,
  recommending safetensors) **and** `varkoder_model.safetensors` + `config.json`
  via `save_varkoder_model`.
- **Fine-tuning input:** `--pretrained-model` handled through `resolve_model`
  (local dir / local `.pkl` with warning / HF repo id). The base architecture is
  recovered from `config["architecture"]`; the body is loaded with
  `strict=False` (head skipped on shape mismatch) — the current behavior.
- **Default body:** `--pretrained-model` defaults to `DEFAULT_MODEL`. A plain
  `varKoder train` fine-tunes from the published model as it does today; the
  `--random-weights` flag (and non-default `--architecture`) opts out to
  scratch/in21k.

### `xtra_scripts/push_to_hf.py` (rewrite — the "homework")

One-time maintainer action to publish the faithful inference weights:
1. Load the trusted local `model.pkl`.
2. `save_varkoder_model` → `varkoder_model.safetensors` + `config.json`
   (with `is_multilabel`).
3. **Verify fidelity:** reload via `resolve_model` + `build_learner`, run both
   the pkl learner and the rebuilt learner on a set of sample images, and assert
   the prediction tensors are `allclose` within tolerance. **This gates the
   push.**
4. Push `varkoder_model.safetensors` + updated `config.json` to HF, leaving the
   legacy files in place.

## Config / CLI / dependencies

- `varKoder/core/config.py`: `DEFAULT_ARCHITECTURE = "vit_large_patch32_224"`;
  keep `DEFAULT_MODEL = "brunoasm/vit_large_patch32_224.NCBI_SRA"`.
- `varKoder/cli.py`: `--architecture` default → new `DEFAULT_ARCHITECTURE`;
  `--pretrained-model` default → `DEFAULT_MODEL`.
- `pyproject.toml`: add `safetensors`. Keep `fastai` and `huggingface_hub`;
  retain the `[fastai]` extra, since the legacy old-style-HF-repo fallback still
  uses `from_pretrained_fastai`.

## Documentation

- `docs/train.md`: document the new export files; add a **deprecation notice**
  that `.pkl` export is deprecated and will be removed in a future release.
- `docs/query.md`: document loading from safetensors (local dir + HF id); note
  legacy `.pkl` still loads with a security warning.
- `README.md`: brief note on the format change and the pkl deprecation where
  models/training/querying are described.

## Testing (TDD)

- **Round-trip fidelity:** train a tiny model → export → reload via the new path
  → predictions identical (`allclose`) to the in-memory learner. Both
  single-label and multi-label.
- **Legacy `.pkl`** loads and emits the security warning.
- **Deprecation warning** emitted on pkl export.
- **Config drives multilabel:** `is_multilabel` selects sigmoid+threshold vs
  softmax+argmax correctly.
- **Fine-tuning** from a safetensors model transfers the body (`strict=False`)
  and produces a fresh head for a changed vocab.
- **HF load** path (small/mocked) resolves and builds.

## Risks / things to watch

- fp16 → fp32 cast on export; ensure contiguous tensors for `safetensors`.
- Building an inference `dls` without training data — pin vocab explicitly so
  head-output order matches `label_names`.
- Custom architectures (`FiannacaModel`, etc.) must round-trip through
  `save`/`build` via the custom-arch `Learner` path.
- The fidelity check in the push script is the guard that the published
  safetensors truly matches the pkl; do not push if it fails.
