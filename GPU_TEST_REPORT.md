# varKoder 1.8.0 GPU validation report

- Commit tested: `793b4f60e798f8fefd410f39b900f153ffaef527` (`test/gpu-1.8.0` = `main` @ `6c2a7c5` + `GPU_TEST_1.8.0.md`)
- Date: 2026-11-23
- Machine: 4× NVIDIA RTX A5000 (24 GB each), driver 575.57.08, CUDA 12.9 (nvidia-smi), Linux 5.15.0-139 (Ubuntu 20.04 HWE kernel). GPUs 2 and 3 were occupied by another user's vLLM job (~21.5 GB each), so single-GPU steps used GPU 0 and the multi-GPU step used GPUs 0,1.
- Environment A: Python 3.11.9 / torch 2.6.0+cu118 (CUDA 11.8, available) / fastai 2.7.19 / timm 1.0.15, built with `conda env create -f conda_environments/linux.yml` (Miniforge, conda 26.9.4)
- Environment B: Python 3.11.9 venv / fastai 2.8.12 / timm 1.0.31, with two torch builds:
  - default `pip install torch torchvision`: torch 2.15.0+cu132 (CUDA **not available**: driver too old for CUDA 13.2)
  - `--index-url https://download.pytorch.org/whl/cu128`: torch 2.11.0+cu128 (CUDA available)

## Results

| Step | Result (PASS / FAIL / SKIPPED) | Notes |
| --- | --- | --- |
| 1a unit tests (default / slow) | PASS | 98 passed / 5 passed |
| 1b integration suite | PASS | all 12 commands succeeded (IM, IM_FASTA, C, IM_STACK, C_STACK, T1, T2, Q1, Q2, Q_STACK, Q_STACK_HF, Q_FASTA); run with 16 cores, not the default 71 (shared server) |
| 1b training used GPU | PASS | summary shows `GPU: 0`; T1 and T2 log "GPU available"; GPU 0 memory 36 → 626 MiB during training, peak util 5% (tiny dataset, 5 s sampling). Suite queries used cpu (8–117 images < 128), as expected |
| 1b T1 config has normalize stats | PASS | `{'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}` |
| 1c(i) default model GPU vs CPU | PASS | 234 rows; cuda then cpu; predicted_labels identical 100.0%; max abs prob diff 5.10e-06; 0 pickle mentions |
| 1c(ii) fine-tune keeps normalization on GPU | PASS | trained on GPU, no device-mismatch error; T1 and fine-tuned `normalize` identical, non-null |
| 1c(iii) GPU-trained model queries on GPU | PASS | 234 images, will use cuda; predictions match actual labels in sampled rows |
| 1c(iv) default fine-tune from published model | PASS | loaded `brunoasm/vit_large_patch32_224.NCBI_SRA`, trained on GPU, no OOM (GPU 0 peak 3.8 GB); config: `vit_large_patch32_224 6 None` |
| 1c(v) multi-GPU | PASS | `CUDA_VISIBLE_DEVICES=0,1`; ran without error, GPU 1 reached 619 MiB during training; `normalize` preserved |
| 2 pip install (fastai 2.8) default query | PASS (with note) | fastai 2.8.12 loads the default model, 0 pickle mentions, predicted_labels 100% identical to env A GPU. With default pip torch (cu132) the query ran on **cpu** because the driver is too old; with cu128 wheels it ran on **cuda**, also 100% identical |

## Failures

None in varKoder.

## TEST SUMMARY from suite.log

```
===== TEST SUMMARY =====
Mode: local conda
Architecture: resnet18
CPU Cores: 16
GPU: 0

Commands completed successfully:
- IM (Image generation from FASTQ)
- IM_FASTA (Image generation from FASTA, 1 Kbp min)
- C (Image conversion)
- IM_STACK (Multi-frame/APNG image generation)
- C_STACK (Multi-frame image conversion)
- T1 (Training with pre-trained weights, 5 epochs)
- T2 (Training from scratch, 5 epochs)
- Q1 (Query from FASTQ)
- Q2 (Query from images)
- Q_STACK (Query from stacked images, --all-frames)
- Q_STACK_HF (Query from stacked images, --all-frames, default model)
- Q_FASTA (Query from FASTA images with default model)

Commands that failed:
None
```

## Anything else unexpected

- **Default pip torch wheels may not use the GPU on older drivers.** `pip install torch` now resolves to a CUDA 13.2 build, which needs a newer driver than 575 (CUDA 12.9); torch then silently falls back to CPU. Not a varKoder bug, but pip users with GPUs may need `--index-url https://download.pytorch.org/whl/cu12x`. Worth a line in the install docs.
- **`conda env create -f linux.yml` hung for >2 h** with an older conda (23.3.1, classic solver, flexible channel priority). With a current conda (libmamba solver) it finished in under 3 minutes. Worth recommending a recent conda/Miniforge in the install docs.
- Warnings seen, all expected or upstream: fastai 2.7.19 + torch 2.6 `torch.cuda.amp` FutureWarnings; varKoder's own `trained_model.pkl` deprecation FutureWarning; "No sample contains more than one label" (test data is single-label); in env B, torch 2.15's `torch.jit.script` FutureWarning; in pytest, the pickle warning from tests that deliberately load legacy `.pkl` models; a pandas `mode.chained_assignment` deprecation from fastai.
