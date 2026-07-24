# Weights-only (safetensors) Model Loading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load varKoder models from safetensors weights + a JSON config instead of pickled fastai `Learner` files, across query, training export, and fine-tuning, with predictions identical to the current pkl and full backward compatibility during a transition.

**Architecture:** A new `varKoder/core/model_io.py` owns the format: `save_varkoder_model` writes `varkoder_model.safetensors` (the fastai `Learner.model.state_dict()`) + `config.json`; `resolve_model` reads them from a local dir, a legacy `.pkl` (with a security warning), or an HF repo id; `build_learner` rebuilds the identical fastai `Learner` and loads the weights. A shared `make_dataloaders` (refactored out of `train_nn`) guarantees inference preprocessing matches training. Custom architectures (which use `LazyLinear`) are relocated to `varKoder/models/custom.py` and materialized at the stored input size before loading weights.

**Tech Stack:** Python 3.11, fastai 2.7.19, timm 1.0.15, huggingface_hub 0.29.x, safetensors 0.6.x, PyTorch, pytest (new dev dependency).

## Global Constraints

- Work inside the existing `varKoder` conda env: `conda activate varKoder`.
- Pinned deps (do not bump): `fastai==2.7.19`, `timm==1.0.15`, `huggingface_hub~=0.29.0`. `safetensors` (0.6.2) is already present via timm; declare it explicitly.
- Python floor: `requires-python = ">=3.11"`.
- Artifact filenames (exact): weights `varkoder_model.safetensors`, config `config.json`.
- `config.json` schema: `{architecture, label_names, is_multilabel, num_classes, input_size, normalize}` where `architecture` is the **base** timm name (never the `hf-hub:` form), `input_size` is `[C, H, W]`, and `normalize` is either `null` (no image normalization) or `{"mean": [...], "std": [...]}`. `normalize` is captured from the source learner's dataloaders at save time and reapplied at load time — fastai's `vision_learner` only adds normalization when `pretrained=True`, so the weights-only path (which uses `pretrained=False`) must carry it explicitly or predictions drift.
- Backward compatibility: legacy `.pkl` still loads (with a `UserWarning` security warning); `train` writes **both** `trained_model.pkl` (with a `DeprecationWarning`) and the safetensors artifact; the HF repo keeps `model.pkl`.
- Predictions from the weights-only path must equal the pkl path within `atol=1e-5` (fidelity is a hard requirement).
- All automated tests run on **CPU** and must not require network access or download pretrained weights (use `pretrained=False`).
- Follow the user's git preference: commit after each task's tests pass.

---

## File Structure

- **Create** `varKoder/models/custom.py` — the custom model classes (`Fiannaca2018Model`, `Arias2022Model` and their Body/Head parts) relocated from `train.py`, plus `instantiate_custom_model(architecture, num_classes, input_size)`. Single responsibility: define + build custom architectures. Imports only torch (no circular import with `train.py`/`model_io.py`).
- **Create** `varKoder/core/model_io.py` — the weights-only format: constants, `save_varkoder_model`, `recover_architecture`, `resolve_model`, `build_learner`. Single responsibility: serialize/deserialize a varKoder model.
- **Create** `varKoder/core/preprocessing.py` — `make_dataloaders(...)` extracted from `train_nn`. Single responsibility: build fastai `DataLoaders` + transforms consistently for train and inference. (Placed in `core` because both `commands/train.py` and `core/model_io.py` import it.)
- **Modify** `varKoder/commands/train.py` — import custom models from `models/custom.py`; call `make_dataloaders`; export both formats with a deprecation warning; resolve `--pretrained-model` via `model_io`.
- **Modify** `varKoder/commands/query.py` — `load_model` uses `resolve_model` + `build_learner`; multilabel from config.
- **Modify** `varKoder/core/config.py` — `DEFAULT_ARCHITECTURE = "vit_large_patch32_224"`.
- **Modify** `varKoder/cli.py` — `--architecture` / `--pretrained-model` / `--model` defaults and help text.
- **Modify** `xtra_scripts/push_to_hf.py` — rewrite to publish the faithful inference artifact with a fidelity gate.
- **Modify** `pyproject.toml`, `conda_environments/mac.yml`, `conda_environments/linux.yml` — add `safetensors` and `pytest`.
- **Modify** `docs/train.md`, `docs/query.md`, `README.md` — document the format and the pkl deprecation.
- **Create** `tests/conftest.py`, `tests/test_*.py` — pytest suite.

---

## Task 1: Test infrastructure and dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `conda_environments/mac.yml`
- Modify: `conda_environments/linux.yml`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`
- Create: `tests/pytest.ini` (or add `[tool.pytest.ini_options]` to `pyproject.toml`)

**Interfaces:**
- Produces: pytest fixtures `synthetic_images` → `(pandas.DataFrame with columns path,labels,is_valid, list[str] labels)`, and `tiny_timm_learner` → a CPU `fastai.learner.Learner` (resnet18, `pretrained=False`, single-label).

- [ ] **Step 1: Add dependencies**

In `pyproject.toml`, under `[project] dependencies`, add `"safetensors"`. Add a test extra:

```toml
[project.optional-dependencies]
test = ["pytest"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

In both `conda_environments/mac.yml` and `conda_environments/linux.yml`, add under the `pip:` list:

```yaml
    - safetensors
    - pytest
```

- [ ] **Step 2: Install pytest into the env**

Run: `conda run -n varKoder python -m pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 3: Write conftest fixtures**

Create `tests/conftest.py`:

```python
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock, ColReader, ColSplitter,
    vision_learner, Resize, ResizeMethod,
)
from fastai.losses import CrossEntropyLossFlat


@pytest.fixture
def synthetic_images(tmp_path):
    """Six 64x64 RGB PNGs across three single labels; 4 train / 2 valid."""
    rng = np.random.default_rng(0)
    labels = ["alpha", "beta", "gamma"]
    rows = []
    for i in range(6):
        arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr).save(p)
        rows.append({"path": str(p), "labels": labels[i % 3], "is_valid": i >= 4})
    return pd.DataFrame(rows), labels


@pytest.fixture
def tiny_timm_learner(synthetic_images):
    """A CPU resnet18 vision_learner with random weights (no download)."""
    df, labels = synthetic_images
    dbl = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=labels)),
        splitter=ColSplitter(),
        get_x=ColReader("path"),
        get_y=ColReader("labels"),
        item_tfms=Resize(64, method=ResizeMethod.Squish),
    )
    dls = dbl.dataloaders(df, bs=2, device="cpu", num_workers=0)
    learn = vision_learner(
        dls, "resnet18", pretrained=False, normalize=True,
        loss_func=CrossEntropyLossFlat(),
    )
    return learn
```

- [ ] **Step 4: Write the smoke test**

Create `tests/test_smoke.py`:

```python
import torch


def test_fixtures_build(tiny_timm_learner, synthetic_images):
    df, labels = synthetic_images
    dl = tiny_timm_learner.dls.test_dl(df)
    preds, _ = tiny_timm_learner.get_preds(dl=dl)
    assert preds.shape == (len(df), len(labels))
    assert torch.isfinite(preds).all()
```

- [ ] **Step 5: Run the smoke test**

Run: `conda run -n varKoder python -m pytest tests/test_smoke.py -v`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml conda_environments/mac.yml conda_environments/linux.yml tests/conftest.py tests/test_smoke.py
git commit -m "test: add pytest infra, fixtures, and safetensors dependency"
```

---

## Task 2: Relocate custom models to `varKoder/models/custom.py`

**Files:**
- Create: `varKoder/models/custom.py`
- Modify: `varKoder/commands/train.py:52-151` (remove the class defs and `build_custom_model` body; import + delegate)
- Create: `tests/test_custom_models.py`

**Interfaces:**
- Produces: `instantiate_custom_model(architecture: str, num_classes: int, input_size) -> torch.nn.Module` where `input_size` is `(C, H, W)`; the returned model has its `LazyLinear` layers materialized (a `state_dict()` with concrete shapes). Also exports classes `Fiannaca2018Model`, `Arias2022Model`.
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write the failing test**

Create `tests/test_custom_models.py`:

```python
import torch
from varKoder.models.custom import instantiate_custom_model


def test_fiannaca_materializes_and_forwards():
    m = instantiate_custom_model("fiannaca2018", num_classes=5, input_size=(3, 64, 64))
    x = torch.randn(2, 3, 64, 64)
    out = m(x)
    assert out.shape == (2, 5)
    # No uninitialized lazy params remain
    for p in m.parameters():
        assert not isinstance(p, torch.nn.parameter.UninitializedParameter)


def test_arias_materializes_and_forwards():
    m = instantiate_custom_model("arias2022", num_classes=3, input_size=(3, 64, 64))
    out = m(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 3)


def test_unknown_arch_raises():
    import pytest
    with pytest.raises(Exception):
        instantiate_custom_model("nope", 2, (3, 64, 64))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n varKoder python -m pytest tests/test_custom_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'varKoder.models.custom'`.

- [ ] **Step 3: Create the module**

Create `varKoder/models/custom.py` (classes moved verbatim from `train.py:52-125`):

```python
"""Custom (non-timm) architectures for varKoder, plus a factory that
materializes their LazyLinear layers so weights can be loaded from safetensors."""

import torch
from torch.nn import (
    Module, Sequential, Linear, Flatten, LazyLinear, ReLU, Dropout,
    Conv1d, MaxPool1d,
)


class Arias2022Head(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.head = Sequential(Linear(64, n_classes))

    def forward(self, x):
        return self.head(x)


class Arias2022Body(Module):
    def __init__(self):
        super().__init__()
        self.body = Sequential(
            Flatten(), LazyLinear(512), ReLU(), Dropout(0.5),
            Linear(512, 64), ReLU(), Dropout(0.5),
        )

    def forward(self, x):
        x = x[:, 0, :, :]
        return self.body(x)


class Fiannaca2018Head(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.head = Sequential(Linear(500, n_classes))

    def forward(self, x):
        return self.head(x)


class Fiannaca2018Body(Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.body = Sequential(
            Conv1d(1, 5, kernel_size=5), ReLU(), MaxPool1d(kernel_size=2),
            Conv1d(5, 10, kernel_size=5), ReLU(), MaxPool1d(kernel_size=2),
            Flatten(), LazyLinear(500), ReLU(),
        )

    def forward(self, x):
        x = x[:, 0, :, :]
        x = self.flatten(x)
        x = x.unsqueeze(1)
        return self.body(x)


class Fiannaca2018Model(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = Sequential(Fiannaca2018Body(), Fiannaca2018Head(n_classes))

    def forward(self, x):
        return self.model(x)


class Arias2022Model(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = Sequential(Arias2022Body(), Arias2022Head(n_classes))

    def forward(self, x):
        return self.model(x)


def instantiate_custom_model(architecture, num_classes, input_size):
    """Build a custom model and materialize its LazyLinear layers with a dummy
    forward at ``input_size`` (C, H, W), so its state_dict has concrete shapes."""
    if architecture == "arias2022":
        model = Arias2022Model(num_classes)
    elif architecture == "fiannaca2018":
        model = Fiannaca2018Model(num_classes)
    else:
        raise Exception("Custom models must be one of: fiannaca2018 arias2022")
    c, h, w = input_size
    with torch.no_grad():
        model(torch.randn(1, c, h, w))
    return model
```

- [ ] **Step 4: Delegate from `train.py`**

In `varKoder/commands/train.py`, delete the custom class definitions (`train.py:52-125`) and add an import near the other varKoder imports (after `train.py:46`):

```python
from varKoder.models.custom import (
    Fiannaca2018Model, Arias2022Model, instantiate_custom_model,
)
```

Replace the body of `build_custom_model` (`train.py:127-151`) with:

```python
def build_custom_model(architecture, dls):
    xb, _ = dls.one_batch()
    input_image_size = xb.shape[-2:]
    return instantiate_custom_model(
        architecture, len(dls.vocab), (1, input_image_size[0], input_image_size[1])
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n varKoder python -m pytest tests/test_custom_models.py -v`
Expected: PASS (3 passed).

Run: `conda run -n varKoder python -c "import varKoder.commands.train"`
Expected: no error (import still works after the refactor).

- [ ] **Step 6: Commit**

```bash
git add varKoder/models/custom.py varKoder/commands/train.py tests/test_custom_models.py
git commit -m "refactor: relocate custom models to models/custom.py with materializing factory"
```

---

## Task 3: Extract `make_dataloaders` shared by train and inference

**Files:**
- Create: `varKoder/core/preprocessing.py`
- Modify: `varKoder/commands/train.py:329-385` (call the extracted function)
- Create: `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `make_dataloaders(df, architecture, is_multilabel, *, bs, device="cpu", num_workers=0, max_lighting=0, p_lighting=0, random_erasing=False, vocab=None) -> fastai DataLoaders`. When `vocab` is given, the category vocab is pinned to it (so head-output order is fixed); otherwise it is inferred from `df`.
- Consumes: `CUSTOM_ARCHS` from `varKoder.core.config`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_preprocessing.py`:

```python
from varKoder.core.preprocessing import make_dataloaders


def test_vocab_is_pinned(synthetic_images):
    df, labels = synthetic_images
    pinned = ["gamma", "beta", "alpha"]  # deliberately not sorted
    dls = make_dataloaders(df, "resnet18", is_multilabel=False, bs=2, vocab=pinned)
    assert list(dls.vocab) == pinned


def test_custom_arch_has_no_resize(synthetic_images):
    df, labels = synthetic_images
    dls = make_dataloaders(df, "fiannaca2018", is_multilabel=False, bs=2, vocab=labels)
    # custom archs use no item resize transform
    assert not any(type(t).__name__ == "Resize" for t in dls.after_item.fs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n varKoder python -m pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'varKoder.core.preprocessing'`.

- [ ] **Step 3: Create the module (logic lifted from `train_nn`)**

Create `varKoder/core/preprocessing.py`:

```python
"""Shared fastai DataLoaders construction, used by training and by
weights-only inference so preprocessing is identical in both paths."""

from PIL.Image import Resampling
from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock, MultiCategoryBlock, ColReader,
    ColSplitter, RandomSplitter, Resize, ResizeMethod, aug_transforms,
    RandomErasing, default_device,
)
from timm import create_model

from varKoder.core.config import CUSTOM_ARCHS


def make_dataloaders(df, architecture, is_multilabel, *, bs, device="cpu",
                     num_workers=0, max_lighting=0, p_lighting=0,
                     random_erasing=False, vocab=None):
    item_transforms = None
    if architecture not in CUSTOM_ARCHS:
        default_cfg = create_model(architecture, pretrained=False).default_cfg
        if default_cfg.get("fixed_input_size"):
            item_transforms = Resize(
                size=default_cfg["input_size"][1:],
                method=ResizeMethod.Squish,
                resamples=(Resampling.BOX, Resampling.BOX),
            )

    transforms = aug_transforms(
        do_flip=False, max_rotate=0, max_zoom=1, max_lighting=max_lighting,
        max_warp=0, p_affine=0, p_lighting=p_lighting,
    )
    if random_erasing:
        transforms.append(RandomErasing())

    if is_multilabel:
        cat_block = MultiCategoryBlock(vocab=vocab) if vocab is not None else MultiCategoryBlock
        blocks = (ImageBlock, cat_block)
        get_y = ColReader("labels", label_delim=";")
    else:
        cat_block = CategoryBlock(vocab=vocab) if vocab is not None else CategoryBlock
        blocks = (ImageBlock, cat_block)
        get_y = ColReader("labels")

    splitter = ColSplitter() if "is_valid" in df.columns else RandomSplitter()

    dbl = DataBlock(
        blocks=blocks, splitter=splitter, get_x=ColReader("path"), get_y=get_y,
        item_tfms=item_transforms, batch_tfms=transforms,
    )
    return dbl.dataloaders(df, bs=bs, device=device, num_workers=num_workers)
```

- [ ] **Step 4: Call it from `train_nn`**

In `varKoder/commands/train.py`, replace the block that builds `item_transforms`, `transforms`, `blocks`, `get_y`, `dbl`, and `dls` (`train.py:329-385`) with a call to `make_dataloaders` (add `from varKoder.core.preprocessing import make_dataloaders` to the imports). Keep the dynamic `batch_size` computation above it:

```python
    device = torch.device('cpu') if force_cpu else default_device()
    dls = make_dataloaders(
        df, architecture, is_multilabel, bs=batch_size, device=device,
        num_workers=num_workers, max_lighting=max_lighting, p_lighting=p_lighting,
        random_erasing=random_erasing,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n varKoder python -m pytest tests/test_preprocessing.py tests/test_smoke.py -v`
Expected: PASS.

Run: `conda run -n varKoder python -c "import varKoder.commands.train"`
Expected: no error.

- [ ] **Step 6: Commit**

```bash
git add varKoder/core/preprocessing.py varKoder/commands/train.py tests/test_preprocessing.py
git commit -m "refactor: extract make_dataloaders shared by train and inference"
```

---

## Task 4: `save_varkoder_model` + `recover_architecture`

**Files:**
- Create: `varKoder/core/model_io.py`
- Create: `tests/test_model_io_save.py`

**Interfaces:**
- Produces:
  - `MODEL_WEIGHTS_FILENAME = "varkoder_model.safetensors"`, `MODEL_CONFIG_FILENAME = "config.json"`.
  - `recover_architecture(learn) -> str` — timm base name via `learn.model[0].model.default_cfg["architecture"]`, else raises (caller passes the name for custom archs).
  - `save_varkoder_model(learn, outdir, *, architecture, is_multilabel) -> None` — writes the two files; state dict cast to fp32 and made contiguous; `input_size` from `learn.dls.one_batch()[0].shape[1:]`; `label_names` from `learn.dls.vocab`.
- Consumes: nothing from later tasks.

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_io_save.py`:

```python
import json
import torch
from safetensors.torch import load_file
from varKoder.core.model_io import (
    save_varkoder_model, MODEL_WEIGHTS_FILENAME, MODEL_CONFIG_FILENAME,
)


def test_save_writes_files_and_config(tiny_timm_learner, tmp_path):
    save_varkoder_model(
        tiny_timm_learner, tmp_path, architecture="resnet18", is_multilabel=False,
    )
    weights = tmp_path / MODEL_WEIGHTS_FILENAME
    config = tmp_path / MODEL_CONFIG_FILENAME
    assert weights.exists() and config.exists()

    cfg = json.loads(config.read_text())
    assert cfg["architecture"] == "resnet18"
    assert cfg["is_multilabel"] is False
    assert cfg["label_names"] == list(tiny_timm_learner.dls.vocab)
    assert cfg["num_classes"] == len(tiny_timm_learner.dls.vocab)
    assert len(cfg["input_size"]) == 3
    assert "normalize" in cfg  # None here (fixture is pretrained=False)

    saved = load_file(str(weights))
    ref = tiny_timm_learner.model.state_dict()
    assert set(saved.keys()) == set(ref.keys())
    for k in ref:
        assert saved[k].dtype == torch.float32


def test_save_captures_normalization(tiny_timm_learner, tmp_path):
    from fastai.vision.all import Normalize
    tiny_timm_learner.dls.add_tfms(
        [Normalize.from_stats([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])], "after_batch")
    save_varkoder_model(tiny_timm_learner, tmp_path,
                        architecture="resnet18", is_multilabel=False)
    cfg = json.loads((tmp_path / MODEL_CONFIG_FILENAME).read_text())
    assert cfg["normalize"] is not None
    assert len(cfg["normalize"]["mean"]) == 3 and len(cfg["normalize"]["std"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n varKoder python -m pytest tests/test_model_io_save.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'varKoder.core.model_io'`.

- [ ] **Step 3: Implement save side**

Create `varKoder/core/model_io.py`:

```python
"""Weights-only (safetensors) serialization for varKoder models."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file
from fastai.vision.all import Normalize

MODEL_WEIGHTS_FILENAME = "varkoder_model.safetensors"
MODEL_CONFIG_FILENAME = "config.json"


def _extract_normalize(dls):
    """Return {'mean':[...], 'std':[...]} for a Normalize in the batch pipeline,
    or None. fastai stores mean/std as (1,C,1,1) tensors."""
    for t in dls.after_batch.fs:
        if isinstance(t, Normalize):
            return {"mean": t.mean.flatten().tolist(),
                    "std": t.std.flatten().tolist()}
    return None


def recover_architecture(learn):
    """Return the base architecture name of a learner (timm or custom)."""
    from varKoder.models.custom import Fiannaca2018Model, Arias2022Model
    if isinstance(learn.model, Fiannaca2018Model):
        return "fiannaca2018"
    if isinstance(learn.model, Arias2022Model):
        return "arias2022"
    return learn.model[0].model.default_cfg["architecture"]


def _fp32_contiguous_state_dict(model):
    return {k: v.detach().to(torch.float32).contiguous()
            for k, v in model.state_dict().items()}


def save_varkoder_model(learn, outdir, *, architecture, is_multilabel):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xb, _ = learn.dls.one_batch()
    input_size = list(xb.shape[1:])  # (C, H, W)
    label_names = list(learn.dls.vocab)

    state_dict = _fp32_contiguous_state_dict(learn.model)
    save_file(state_dict, str(outdir / MODEL_WEIGHTS_FILENAME))

    config = {
        "architecture": architecture,
        "label_names": label_names,
        "is_multilabel": bool(is_multilabel),
        "num_classes": len(label_names),
        "input_size": input_size,
        "normalize": _extract_normalize(learn.dls),
    }
    (outdir / MODEL_CONFIG_FILENAME).write_text(json.dumps(config, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n varKoder python -m pytest tests/test_model_io_save.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add varKoder/core/model_io.py tests/test_model_io_save.py
git commit -m "feat: save varKoder models as safetensors + config.json"
```

---

## Task 5: `build_learner` and round-trip fidelity

**Files:**
- Modify: `varKoder/core/model_io.py`
- Create: `tests/test_model_io_roundtrip.py`

**Interfaces:**
- Produces: `build_learner(config: dict, device="cpu") -> fastai Learner` — a learner whose model matches the trained architecture; for timm archs uses `vision_learner(pretrained=False)`; for custom archs uses `instantiate_custom_model` + a bare `Learner`. Vocab pinned to `config["label_names"]`.
- Consumes: `make_dataloaders` (Task 3), `instantiate_custom_model` (Task 2), `save_varkoder_model` (Task 4).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model_io_roundtrip.py`:

```python
import json
import torch
from safetensors.torch import load_file
from varKoder.core.model_io import (
    save_varkoder_model, build_learner, MODEL_CONFIG_FILENAME, MODEL_WEIGHTS_FILENAME,
)


def _preds(learn, df):
    dl = learn.dls.test_dl(df)
    p, _ = learn.get_preds(dl=dl)
    return p


def test_timm_roundtrip_fidelity(tiny_timm_learner, synthetic_images, tmp_path):
    df, _ = synthetic_images
    baseline = _preds(tiny_timm_learner, df)

    save_varkoder_model(tiny_timm_learner, tmp_path,
                        architecture="resnet18", is_multilabel=False)
    cfg = json.loads((tmp_path / MODEL_CONFIG_FILENAME).read_text())
    state = load_file(str(tmp_path / MODEL_WEIGHTS_FILENAME))

    learn2 = build_learner(cfg, device="cpu")
    learn2.model.load_state_dict(state, strict=True)
    after = _preds(learn2, df)

    assert torch.allclose(baseline, after, atol=1e-5)


def test_normalized_model_roundtrip_fidelity(tiny_timm_learner, synthetic_images, tmp_path):
    """A model trained WITH normalization must still round-trip exactly. This is
    the case fastai's pretrained=False path would silently drop, so it guards
    that build_learner reapplies normalization from config."""
    from fastai.vision.all import Normalize
    learn = tiny_timm_learner
    # cuda=False keeps stats on CPU so this runs on any machine (mps/cuda
    # available would otherwise put stats off-device and mismatch the CPU batch).
    learn.dls.add_tfms(
        [Normalize.from_stats([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], cuda=False)],
        "after_batch")
    df, _ = synthetic_images
    baseline = _preds(learn, df)

    save_varkoder_model(learn, tmp_path, architecture="resnet18", is_multilabel=False)
    cfg = json.loads((tmp_path / MODEL_CONFIG_FILENAME).read_text())
    assert cfg["normalize"] is not None
    state = load_file(str(tmp_path / MODEL_WEIGHTS_FILENAME))

    learn2 = build_learner(cfg, device="cpu")
    learn2.model.load_state_dict(state, strict=True)
    after = _preds(learn2, df)

    assert torch.allclose(baseline, after, atol=1e-5)


def test_custom_roundtrip_builds(synthetic_images, tmp_path):
    # config for a custom arch; weights come from a freshly built learner
    from varKoder.core.preprocessing import make_dataloaders
    from fastai.vision.all import Learner
    from fastai.losses import CrossEntropyLossFlat
    from varKoder.models.custom import instantiate_custom_model

    df, labels = synthetic_images
    dls = make_dataloaders(df, "fiannaca2018", is_multilabel=False, bs=2, vocab=labels)
    model = instantiate_custom_model("fiannaca2018", len(labels), (1, 64, 64))
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat())
    save_varkoder_model(learn, tmp_path, architecture="fiannaca2018", is_multilabel=False)

    cfg = json.loads((tmp_path / MODEL_CONFIG_FILENAME).read_text())
    state = load_file(str(tmp_path / MODEL_WEIGHTS_FILENAME))
    learn2 = build_learner(cfg, device="cpu")
    learn2.model.load_state_dict(state, strict=True)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n varKoder python -m pytest tests/test_model_io_roundtrip.py -v`
Expected: FAIL with `AttributeError`/`ImportError: cannot import name 'build_learner'`.

- [ ] **Step 3: Implement `build_learner`**

Add to `varKoder/core/model_io.py` (imports at top of file):

```python
import numpy as np
import pandas as pd
from PIL import Image
from fastai.vision.all import vision_learner, Learner
from fastai.losses import CrossEntropyLossFlat

from varKoder.core.config import CUSTOM_ARCHS
from varKoder.core.preprocessing import make_dataloaders
from varKoder.models.custom import instantiate_custom_model


def _dummy_df(label_names, input_size, tmpdir):
    """Two placeholder rows (one train, one valid) so a DataLoaders can be built
    without the original training data. Vocab is pinned separately."""
    c, h, w = input_size
    rows = []
    for i in range(2):
        arr = (np.zeros((h, w, 3))).astype("uint8")
        p = Path(tmpdir) / f"_dummy_{i}.png"
        Image.fromarray(arr).save(p)
        rows.append({"path": str(p), "labels": label_names[0], "is_valid": i == 1})
    return pd.DataFrame(rows)


def build_learner(config, device="cpu"):
    import tempfile
    architecture = config["architecture"]
    label_names = config["label_names"]
    is_multilabel = config["is_multilabel"]
    input_size = tuple(config["input_size"])

    with tempfile.TemporaryDirectory() as tmp:
        df = _dummy_df(label_names, input_size, tmp)
        dls = make_dataloaders(df, architecture, is_multilabel, bs=2,
                               device=device, num_workers=0, vocab=label_names)

        if architecture in CUSTOM_ARCHS:
            model = instantiate_custom_model(architecture, len(label_names), input_size)
            learn = Learner(dls, model, loss_func=CrossEntropyLossFlat())
        else:
            # pretrained=False so no weights download; normalization is NOT added
            # by fastai in this mode, so we reapply it from config below.
            learn = vision_learner(dls, architecture, pretrained=False,
                                   normalize=False, loss_func=CrossEntropyLossFlat())

        norm = config.get("normalize")
        if norm is not None:
            from fastai.vision.all import Normalize
            # from_stats(cuda=True) puts stats on the DEFAULT device via to_device;
            # Normalize.encodes does (x - mean) with no device coercion, so stats
            # must sit on THIS learner's device. Build on CPU, then move explicitly.
            norm_tfm = Normalize.from_stats(norm["mean"], norm["std"], cuda=False)
            norm_tfm.mean = norm_tfm.mean.to(device)
            norm_tfm.std = norm_tfm.std.to(device)
            learn.dls.add_tfms([norm_tfm], "after_batch")
    learn.model = learn.model.to(device)
    return learn
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n varKoder python -m pytest tests/test_model_io_roundtrip.py -v`
Expected: PASS (2 passed). If the timm fidelity test fails on transform differences, verify the fixture and `make_dataloaders` build the same `item_tfms`/`normalize`; they must match.

- [ ] **Step 5: Commit**

```bash
git add varKoder/core/model_io.py tests/test_model_io_roundtrip.py
git commit -m "feat: rebuild fastai Learner from safetensors + config (round-trip fidelity)"
```

---

## Task 6: `resolve_model` (local dir / legacy pkl / HF repo)

**Files:**
- Modify: `varKoder/core/model_io.py`
- Create: `tests/test_model_io_resolve.py`

**Interfaces:**
- Produces: `resolve_model(source: str) -> tuple[dict, dict]` returning `(state_dict, config)`. Resolution order: existing local directory with the two files → read; path ending `.pkl` (or an existing file) → **`UserWarning`** security warning, `load_learner`, extract state dict + synthesize config; otherwise treat as HF repo id → `hf_hub_download` the two files; on `EntryNotFoundError` fall back to legacy `from_pretrained_fastai` (with the security warning). Config synthesis from a learner uses `recover_architecture` (or the custom name if present) and `"MultiLabel" in str(learn.loss_func)`.
- Consumes: `save_varkoder_model` fields, `recover_architecture`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model_io_resolve.py`:

```python
import pytest
from varKoder.core.model_io import save_varkoder_model, resolve_model


def test_resolve_local_dir(tiny_timm_learner, tmp_path):
    save_varkoder_model(tiny_timm_learner, tmp_path,
                        architecture="resnet18", is_multilabel=False)
    state, cfg = resolve_model(str(tmp_path))
    assert cfg["architecture"] == "resnet18"
    assert set(state.keys()) == set(tiny_timm_learner.model.state_dict().keys())


def test_resolve_legacy_pkl_warns(tiny_timm_learner, tmp_path):
    pkl = tmp_path / "trained_model.pkl"
    tiny_timm_learner.export(pkl)
    with pytest.warns(UserWarning):
        state, cfg = resolve_model(str(pkl))
    assert "architecture" in cfg and "label_names" in cfg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n varKoder python -m pytest tests/test_model_io_resolve.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_model'`.

- [ ] **Step 3: Implement `resolve_model`**

Add to `varKoder/core/model_io.py`:

```python
import warnings
from fastai.learner import load_learner

_PICKLE_WARNING = (
    "Loading a pickled (.pkl) model executes arbitrary code on load. This is "
    "deprecated and unsafe; re-export to safetensors with a recent varKoder."
)


def _config_from_learner(learn, architecture=None):
    label_names = list(learn.dls.vocab)
    xb, _ = learn.dls.one_batch()
    if architecture is None:
        architecture = recover_architecture(learn)
    return {
        "architecture": architecture,
        "label_names": label_names,
        "is_multilabel": "MultiLabel" in str(learn.loss_func),
        "num_classes": len(label_names),
        "input_size": list(xb.shape[1:]),
        "normalize": _extract_normalize(learn.dls),
    }


def _read_local_dir(d):
    d = Path(d)
    config = json.loads((d / MODEL_CONFIG_FILENAME).read_text())
    state = load_file(str(d / MODEL_WEIGHTS_FILENAME))
    return state, config


def resolve_model(source):
    p = Path(source)

    if p.is_dir() and (p / MODEL_WEIGHTS_FILENAME).exists():
        return _read_local_dir(p)

    if p.is_file() or str(source).endswith(".pkl"):
        warnings.warn(_PICKLE_WARNING, UserWarning)
        learn = load_learner(source, cpu=True)
        return learn.model.state_dict(), _config_from_learner(learn)

    # Treat as a Hugging Face repo id.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        cfg_path = hf_hub_download(source, MODEL_CONFIG_FILENAME)
        wts_path = hf_hub_download(source, MODEL_WEIGHTS_FILENAME)
        config = json.loads(Path(cfg_path).read_text())
        state = load_file(wts_path)
        return state, config
    except EntryNotFoundError:
        warnings.warn(_PICKLE_WARNING, UserWarning)
        from huggingface_hub import from_pretrained_fastai
        learn = from_pretrained_fastai(source)
        return learn.model.state_dict(), _config_from_learner(learn)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n varKoder python -m pytest tests/test_model_io_resolve.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add varKoder/core/model_io.py tests/test_model_io_resolve.py
git commit -m "feat: resolve models from local dir, legacy pkl, or HF repo"
```

---

## Task 7: Query loads from safetensors

**Files:**
- Modify: `varKoder/commands/query.py:41-43` (imports), `query.py:180-221` (`load_model`), `query.py:289` (multilabel check)
- Create: `tests/test_query_load.py`

**Interfaces:**
- Consumes: `resolve_model`, `build_learner` (Tasks 5-6).
- Produces: `QueryCommand.load_model()` returns a ready `Learner`; multilabel decided by the resolved config, stored as `self.is_multilabel`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_query_load.py`:

```python
import json
import torch
from types import SimpleNamespace
from varKoder.core.model_io import save_varkoder_model
from varKoder.commands.query import QueryCommand


def test_query_loads_local_dir(tiny_timm_learner, synthetic_images, tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    save_varkoder_model(tiny_timm_learner, model_dir,
                        architecture="resnet18", is_multilabel=False)

    args = SimpleNamespace(model=str(model_dir), max_batch_size=2)
    qc = QueryCommand.__new__(QueryCommand)  # bypass __init__
    qc.args = args
    qc.images_d = tmp_path  # no images needed for load_model path
    monkeypatch.setattr(
        "varKoder.commands.query.torch.backends.mps.is_built", lambda: False)
    learn = qc.load_model()
    assert learn.dls.vocab is not None
    assert qc.is_multilabel is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n varKoder python -m pytest tests/test_query_load.py -v`
Expected: FAIL (old `load_model` calls `load_learner`/`from_pretrained_fastai` and has no `is_multilabel`).

- [ ] **Step 3: Rewrite `load_model`**

In `varKoder/commands/query.py`, replace the fastai/hub imports (`query.py:41-43`) with:

```python
from torch import nn
from varKoder.core.model_io import resolve_model, build_learner
```

Replace `load_model` (`query.py:180-221`) with:

```python
    def load_model(self):
        """Load the model for inference from safetensors weights (or legacy pkl).

        Preserves the prior device heuristic: use a GPU only when one is
        available AND there are enough images (>=128) to be worth it.
        """
        n_images = len([img for img in self.images_d.rglob("*.png")])
        gpu_available = torch.backends.mps.is_built() or (
            torch.backends.cuda.is_built() and torch.cuda.device_count()
        )
        if gpu_available:
            eprint("GPU available. Will try to use GPU for processing.")
        else:
            eprint("GPU not available. Using CPU for processing.")

        if gpu_available and n_images >= 128:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = "cpu"
        eprint(n_images, "images in the input, will use", device, "for prediction.")

        state_dict, config = resolve_model(self.args.model)
        self.is_multilabel = config["is_multilabel"]
        learn = build_learner(config, device=device)
        learn.model.load_state_dict(state_dict, strict=True)
        learn.dls.device = device
        return learn
```

Change the prediction branch in `run` (`query.py:289`) from
`if "MultiLabel" in str(learn.loss_func):` to `if self.is_multilabel:`.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n varKoder python -m pytest tests/test_query_load.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add varKoder/commands/query.py tests/test_query_load.py
git commit -m "feat: query loads model from safetensors; multilabel from config"
```

---

## Task 8: Train exports both formats with a deprecation warning

**Files:**
- Modify: `varKoder/commands/train.py:778-787` (the save block)
- Create: `tests/test_train_export.py`

**Interfaces:**
- Consumes: `save_varkoder_model` (Task 4).
- Produces: `train` output dir contains `trained_model.pkl`, `varkoder_model.safetensors`, `config.json`, `labels.txt`, `input_data.csv`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_export.py`:

```python
import warnings
from varKoder.commands.train import export_trained_model  # new helper
from varKoder.core.model_io import MODEL_WEIGHTS_FILENAME, MODEL_CONFIG_FILENAME


def test_export_writes_both_and_warns(tiny_timm_learner, tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_trained_model(
            tiny_timm_learner, tmp_path,
            architecture="resnet18", is_multilabel=False,
        )
    assert (tmp_path / "trained_model.pkl").exists()
    assert (tmp_path / MODEL_WEIGHTS_FILENAME).exists()
    assert (tmp_path / MODEL_CONFIG_FILENAME).exists()
    assert (tmp_path / "labels.txt").exists()
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n varKoder python -m pytest tests/test_train_export.py -v`
Expected: FAIL with `ImportError: cannot import name 'export_trained_model'`.

- [ ] **Step 3: Add the export helper and call it**

In `varKoder/commands/train.py`, add `from varKoder.core.model_io import save_varkoder_model, recover_architecture` to the imports, and add a module-level helper. It normalizes an `hf-hub:` architecture string to the base name (the safetensors config must store the base name) by recovering it from the learner:

```python
def export_trained_model(learn, outdir, *, architecture, is_multilabel):
    """Write both the legacy pkl (deprecated) and the safetensors artifact."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    warnings.warn(
        "Exporting trained_model.pkl is deprecated and will be removed in a "
        "future release; use the safetensors artifact (varkoder_model.safetensors "
        "+ config.json).",
        DeprecationWarning, stacklevel=2,
    )
    # config must store the base timm name, never the "hf-hub:" form
    if architecture.startswith("hf-hub:"):
        architecture = recover_architecture(learn)
    learn.export(outdir / "trained_model.pkl")
    save_varkoder_model(learn, outdir, architecture=architecture,
                        is_multilabel=is_multilabel)
    with open(outdir / "labels.txt", "w") as f:
        f.write("\n".join(learn.dls.vocab))
```

Replace the save block in `TrainCommand.run` (`train.py:778-787`) with:

```python
        outdir = Path(self.args.outdir)
        export_trained_model(
            learn, outdir,
            architecture=train_architecture,
            is_multilabel=not self.args.single_label,
        )
        image_files.to_csv(outdir / "input_data.csv", index=False)
        eprint("Model, labels, and data table saved to directory", str(outdir))
```

`train_architecture` passed in may be the `hf-hub:` form (the old default), a base timm name, or a custom name; `export_trained_model` normalizes the `hf-hub:` case to the base name via `recover_architecture`. Custom names pass through unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n varKoder python -m pytest tests/test_train_export.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add varKoder/commands/train.py tests/test_train_export.py
git commit -m "feat: train exports safetensors + pkl (deprecated) with config"
```

---

## Task 9: Reroute default training body and update CLI defaults

**Files:**
- Modify: `varKoder/core/config.py:51` (`DEFAULT_ARCHITECTURE`)
- Modify: `varKoder/cli.py:221-229` (`--architecture`, `--pretrained-model`), `cli.py:356-361` (`--model` help)
- Modify: `varKoder/commands/train.py:682-694` (pretrained-model handling via `resolve_model`)
- Create: `tests/test_cli_defaults.py`

**Interfaces:**
- Consumes: `resolve_model` (Task 6), `DEFAULT_MODEL`.
- Produces: `--pretrained-model` defaults to `DEFAULT_MODEL`; `--architecture` defaults to `"vit_large_patch32_224"`; train loads the pretrained body's state dict + base architecture through `resolve_model`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_defaults.py`:

```python
from varKoder.cli import setup_parser
from varKoder.core.config import DEFAULT_MODEL


def test_train_defaults():
    # setup_parser() (cli.py:37) returns the configured ArgumentParser;
    # `train` takes positionals `input` and `outdir` (cli.py:178-181).
    args = setup_parser().parse_args(["train", "in", "out"])
    assert args.architecture == "vit_large_patch32_224"
    assert args.pretrained_model == DEFAULT_MODEL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n varKoder python -m pytest tests/test_cli_defaults.py -v`
Expected: FAIL (defaults still the old values).

- [ ] **Step 3: Update config and CLI**

In `varKoder/core/config.py:51`:

```python
DEFAULT_ARCHITECTURE = "vit_large_patch32_224"
```

In `varKoder/cli.py`, `--pretrained-model` (`cli.py:225-229`):

```python
    parser_train.add_argument(
        "-m", "--pretrained-model",
        help=("model to fine-tune from: a local model directory, a Hugging Face "
              "repo id, or a legacy .pkl file (deprecated). Defaults to the "
              "published varKoder model. Pass 'none' to train from --architecture "
              "instead (timm pretrained weights, or random with --random-weights)."),
        default=DEFAULT_MODEL,
    )
```

Add `DEFAULT_MODEL` to the config import in `cli.py` (it already imports many `DEFAULT_*`; add `DEFAULT_MODEL` if missing). Update `--model` help (`cli.py:356-361`):

```python
        help=("trained model: a local model directory (varkoder_model.safetensors "
              "+ config.json), a Hugging Face repo id, or a legacy .pkl file "
              "(deprecated)."),
```

- [ ] **Step 4: Route pretrained-model through `resolve_model`**

In `varKoder/commands/train.py`, add `from varKoder.core.model_io import resolve_model, recover_architecture` to imports. Replace the `if self.args.pretrained_model:` block (`train.py:682-694`) with:

```python
            use_pretrained = (
                self.args.pretrained_model
                and str(self.args.pretrained_model).lower() != "none"
                and not self.args.random_weights
            )
            if use_pretrained:
                eprint("Loading pretrained model from:", str(self.args.pretrained_model))
                pre_state, pre_config = resolve_model(self.args.pretrained_model)
                model_state_dict = pre_state
                train_architecture = pre_config["architecture"]
                pretrained = False
```

The existing downstream code (`train.py:696-704`) already handles the `elif`/`else`: when `--pretrained-model none` is given (or `--random-weights`), `use_pretrained` is False, so the `elif not self.args.random_weights and architecture not in CUSTOM_ARCHS` branch sets `pretrained=True` (timm in21k weights for `--architecture`), and `--random-weights` forces the random `else`. The body is loaded with `strict=False` later (`train.py:434-442`), so a head sized for a new vocab is skipped — unchanged behavior. (`self.args.random_weights` already exists: `train.py:696`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n varKoder python -m pytest tests/test_cli_defaults.py -v`
Expected: PASS.

Run: `conda run -n varKoder python -m pytest tests/ -v`
Expected: PASS (full suite green).

- [ ] **Step 6: Commit**

```bash
git add varKoder/core/config.py varKoder/cli.py varKoder/commands/train.py tests/test_cli_defaults.py
git commit -m "feat: default training body from DEFAULT_MODEL; base architecture default"
```

---

## Task 10: Rewrite `xtra_scripts/push_to_hf.py` with a fidelity gate

**Files:**
- Modify: `xtra_scripts/push_to_hf.py`
- Create: `tests/test_push_fidelity.py`

**Interfaces:**
- Consumes: `save_varkoder_model`, `resolve_model`, `build_learner`.
- Produces: `verify_fidelity(pkl_path, sample_image_dir) -> bool` (importable, testable) and a `main()` that saves + verifies + pushes.

- [ ] **Step 1: Write the failing test**

Create `tests/test_push_fidelity.py` (tests the local save+verify analog, no network):

```python
import json
import torch
from safetensors.torch import load_file
from varKoder.core.model_io import (
    save_varkoder_model, build_learner, resolve_model, MODEL_CONFIG_FILENAME,
)


def test_saved_model_matches_source(tiny_timm_learner, synthetic_images, tmp_path):
    df, _ = synthetic_images
    dl = tiny_timm_learner.dls.test_dl(df)
    baseline, _ = tiny_timm_learner.get_preds(dl=dl)

    save_varkoder_model(tiny_timm_learner, tmp_path,
                        architecture="resnet18", is_multilabel=False)
    state, cfg = resolve_model(str(tmp_path))
    learn2 = build_learner(cfg, device="cpu")
    learn2.model.load_state_dict(state, strict=True)
    after, _ = learn2.get_preds(dl=learn2.dls.test_dl(df))

    assert torch.allclose(baseline, after, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `conda run -n varKoder python -m pytest tests/test_push_fidelity.py -v`
Expected: PASS (this exercises existing functions; it guards the fidelity contract the push script relies on). If it fails, fix `build_learner`/`save_varkoder_model` before proceeding.

- [ ] **Step 3: Rewrite the push script**

Replace `xtra_scripts/push_to_hf.py` with:

```python
#!/usr/bin/env python
"""Publish a varKoder model to Hugging Face as safetensors weights + config.json.

Loads a trusted local .pkl, exports the safetensors artifact, verifies that the
rebuilt model reproduces the pkl's predictions on sample images, and only then
uploads. Legacy files already in the repo are left untouched.
"""

import argparse
import tempfile
from pathlib import Path

import torch
from fastai.vision.all import load_learner
from huggingface_hub import HfApi

from varKoder.core.model_io import (
    save_varkoder_model, build_learner, resolve_model, recover_architecture,
    MODEL_WEIGHTS_FILENAME, MODEL_CONFIG_FILENAME,
)


def verify_fidelity(learn, sample_image_dir, out_dir, atol=1e-4):
    import pandas as pd
    imgs = [str(p) for p in Path(sample_image_dir).rglob("*.png")]
    if not imgs:
        raise SystemExit("No sample PNGs found for fidelity check.")
    df = pd.DataFrame({"path": imgs})
    baseline, _ = learn.get_preds(dl=learn.dls.test_dl(df))

    state, cfg = resolve_model(str(out_dir))
    rebuilt = build_learner(cfg, device="cpu")
    rebuilt.model.load_state_dict(state, strict=True)
    after, _ = rebuilt.get_preds(dl=rebuilt.dls.test_dl(df))
    return torch.allclose(baseline, after, atol=atol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", help="Path to the trusted local .pkl")
    ap.add_argument("repo_id", help="Target Hugging Face repo id")
    ap.add_argument("sample_images", help="Dir of sample PNGs for the fidelity check")
    args = ap.parse_args()

    learn = load_learner(args.model_path, cpu=True)
    architecture = recover_architecture(learn)
    is_multilabel = "MultiLabel" in str(learn.loss_func)

    with tempfile.TemporaryDirectory() as out:
        save_varkoder_model(learn, out, architecture=architecture,
                            is_multilabel=is_multilabel)
        if not verify_fidelity(learn, args.sample_images, out):
            raise SystemExit("Fidelity check FAILED — not pushing.")
        api = HfApi()
        for fname in (MODEL_WEIGHTS_FILENAME, MODEL_CONFIG_FILENAME):
            api.upload_file(path_or_fileobj=str(Path(out) / fname),
                            path_in_repo=fname, repo_id=args.repo_id)
    print("Pushed", MODEL_WEIGHTS_FILENAME, "and", MODEL_CONFIG_FILENAME)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n varKoder python -m pytest tests/test_push_fidelity.py -v`
Expected: PASS.

Run: `conda run -n varKoder python -c "import ast; ast.parse(open('xtra_scripts/push_to_hf.py').read())"`
Expected: no error (script parses).

- [ ] **Step 5: Commit**

```bash
git add xtra_scripts/push_to_hf.py tests/test_push_fidelity.py
git commit -m "feat: push_to_hf publishes safetensors + config with a fidelity gate"
```

> **Manual maintainer step (not automated):** run `conda run -n varKoder python xtra_scripts/push_to_hf.py <local model.pkl> brunoasm/vit_large_patch32_224.NCBI_SRA <sample_images_dir>` to publish the real inference artifact. Confirm the fidelity check passes before it uploads. Leave `model.pkl` on the repo for backward compatibility.

---

## Task 11: Documentation

**Files:**
- Modify: `docs/train.md`
- Modify: `docs/query.md`
- Modify: `README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Update `docs/train.md`**

Add a section describing the export files (`varkoder_model.safetensors`, `config.json`, plus the existing `labels.txt`, `input_data.csv`, and the still-written `trained_model.pkl`). Add a callout:

```markdown
> **Deprecation:** `trained_model.pkl` is still written for backward
> compatibility but is **deprecated** and will be removed in a future release.
> Prefer `varkoder_model.safetensors` + `config.json`, which load without
> executing pickled code.
```

Document that `--pretrained-model` now accepts a local model directory, a Hugging
Face repo id, or a legacy `.pkl`, and defaults to the published model.

- [ ] **Step 2: Update `docs/query.md`**

Document that `--model` accepts a local model directory (safetensors + config), a
Hugging Face repo id, or a legacy `.pkl` (deprecated, loads with a security
warning). Note the default model is loaded as weights-only safetensors.

- [ ] **Step 3: Update `README.md`**

Where models/training/querying are described, add a short note: models are now
distributed as safetensors weights + `config.json`; `.pkl` is deprecated and
loads with a security warning.

- [ ] **Step 4: Verify the deprecation notice landed**

Run: `grep -il "deprecat" docs/train.md docs/query.md README.md`
Expected: all three files listed (each mentions the pkl deprecation).

- [ ] **Step 5: Commit**

```bash
git add docs/train.md docs/query.md README.md
git commit -m "docs: document safetensors model format and pkl deprecation"
```

---

## Final verification

- [ ] Run the full suite: `conda run -n varKoder python -m pytest tests/ -v` — all green.
- [ ] Import sanity: `conda run -n varKoder python -c "import varKoder.cli, varKoder.commands.train, varKoder.commands.query, varKoder.core.model_io"`.
- [ ] Confirm no code path other than the legacy fallback calls `load_learner`/`from_pretrained_fastai` without emitting the security warning.
- [ ] Bump the version in `pyproject.toml` before release (per repo convention: the Docker release workflow tags the image by this version).

## Self-review notes (spec coverage)

- Format (`varkoder_model.safetensors` + `config.json`, fp32, input_size): Tasks 4-5.
- Rebuild fastai Learner, fidelity: Task 5.
- resolve_model (dir / pkl+warning / HF + legacy fallback): Task 6.
- Query weights-only + multilabel-from-config: Task 7.
- Train exports both + pkl deprecation warning: Task 8.
- Default `--pretrained-model = DEFAULT_MODEL`, `--architecture` base, body via resolve_model: Task 9.
- Homework (push script + fidelity gate): Task 10 (+ manual publish step).
- Custom archs (LazyLinear materialization, shared module): Tasks 2, 5.
- Shared preprocessing (single source of truth): Task 3.
- Docs + deprecation notices: Task 11.
- Dependencies (safetensors, pytest) and env files: Task 1.
