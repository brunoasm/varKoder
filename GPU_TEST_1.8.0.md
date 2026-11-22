# GPU validation of varKoder 1.8.0: instructions for an agent

You are validating varKoder 1.8.0 on a machine with an NVIDIA GPU before it is
released. Everything else has already been tested on CPU only, so the point of
this run is the GPU code paths, plus the integration suite against real SRA data.

**Ground rules**

- Test the code on this branch, which is `main` at the 1.8.0 release candidate plus
  this file. **Do not change any varKoder code or tests.** If something fails,
  record it (command, full error, log excerpt) and keep going with the remaining
  steps where possible.
- Write your findings to `GPU_TEST_REPORT.md` at the repository root, using the
  template at the end of this file. Commit only that file to this branch
  (`test/gpu-1.8.0`) and push it. If you cannot push, print the report instead.
- Work in a scratch directory outside the repository for all data and outputs
  (for example `$HOME/varkoder-gpu-test`), so nothing large gets committed.
- Record exact versions: `nvidia-smi`, Python, torch, fastai, timm, CUDA as seen by
  torch, and the commit you tested (`git rev-parse HEAD`).

Expected total time: about 1 hour, most of it the integration suite.

---

## 0. Preflight

```bash
nvidia-smi                       # must list at least one GPU; stop and report if not
git rev-parse HEAD
REPO=$(pwd)                      # repository root
WORK=$HOME/varkoder-gpu-test && mkdir -p "$WORK"
```

The machine needs internet access to NCBI SRA (for `fastq-dump`), Hugging Face
(`huggingface.co` and its CDN, for the default model and timm weights) and the
conda/pip package indexes.

## 1. Environment A: the documented conda environment

This pins torch 2.6.0 (CUDA 11.8), fastai 2.7.19 and timm 1.0.15, plus
sra-tools, dsk, fastp and bbmap.

```bash
conda env create -n varkoder-gpu -f "$REPO/conda_environments/linux.yml"
conda activate varkoder-gpu
pip install -e "$REPO"
python -c "import torch, fastai, timm; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), fastai.__version__, timm.__version__)"
varKoder --version               # must print 1.8.0
```

`torch.cuda.is_available()` must be `True`. If it is `False`, the CUDA 11.8 wheels
probably do not match the driver; report the `nvidia-smi` driver/CUDA version and
stop.

### 1a. Unit tests (sanity check, mostly CPU)

```bash
cd "$REPO" && pytest -q && pytest -q -m slow
```

Expected: 98 passed (default), 5 passed (slow).

### 1b. Integration suite with real SRA data, on the GPU

The suite is interactive. Its prompts, in order, are: test mode, GPU index (asked
only when a GPU is detected), time profiling, architecture, pretraining epochs,
fine-tuning epochs, CPU cores. Answer them by piping input: local conda install,
GPU 0, no profiling, resnet18, 5 + 5 epochs, default cores.

```bash
cd "$WORK"
cp "$REPO"/tests/0*.sh "$REPO"/tests/Bembidion.csv .
bash 01_download_fastqs.sh 2>&1 | tee download.log
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > gpu_usage.csv &
GPU_MON=$!
printf '1\n0\nN\n1\n5\n5\n\n' | bash 03_test_installation.sh 2>&1 | tee suite.log
kill $GPU_MON
```

Check and report:

- The `TEST SUMMARY` at the end of `suite.log`: every command should be listed as
  successful (IM, IM_FASTA, C, IM_STACK, C_STACK, T1, T2, Q1, Q2, Q_STACK,
  Q_STACK_HF, Q_FASTA). Copy the summary into the report.
- The summary should show `GPU: 0`, and the training steps (T1, T2) should log
  `GPU available. Will try to use GPU for processing.` `gpu_usage.csv` should show
  utilization and memory use during training.
- Query steps print `N images in the input, will use <device> for prediction.`
  By design, `query` only uses the GPU for 128 or more images, so the suite's query
  steps are **expected to use cpu**. That is not a failure; step 1c covers GPU
  queries.
- `trained_pretrained/varkoder_config.json` exists. T1 trains resnet18 from timm
  ImageNet weights, so its `"normalize"` should be a dict of mean/std lists, not
  `null`.

Keep `$WORK` (in particular `images/` and `trained_pretrained/`) for the next steps.

### 1c. GPU-specific checks the suite does not cover

Run from `$WORK`, still in environment A.

**(i) Default model, GPU vs CPU.** `images/` holds well over 128 varKodes, so this
query runs on the GPU. The same query with the GPU hidden must give the same
answers.

```bash
varKoder query -I --include-probs images q_default_gpu 2>&1 | tee q_default_gpu.log
CUDA_VISIBLE_DEVICES="" varKoder query -I --include-probs images q_default_cpu 2>&1 | tee q_default_cpu.log
grep "will use" q_default_gpu.log q_default_cpu.log      # expect cuda, then cpu
grep -ci pickle q_default_gpu.log                          # expect 0: loads safetensors, no pickle warning
python - <<'EOF'
import pandas as pd
g = pd.read_csv("q_default_gpu/predictions.csv"); c = pd.read_csv("q_default_cpu/predictions.csv")
assert list(g["varKode_image_path"]) == list(c["varKode_image_path"])
same = (g["predicted_labels"].fillna("") == c["predicted_labels"].fillna("")).mean()
probs = [col for col in c.columns if ":" in col]   # label columns are "key:value"
diff = (g[probs] - c[probs]).abs().to_numpy().max()
print(f"rows={len(g)}  predicted_labels identical: {same:.1%}  max |prob diff| = {diff:.2e}")
EOF
```

Expected: `predicted_labels` identical for 100% of rows and max probability
difference well below 1e-2 (small floating-point differences between GPU and CPU
are normal). Report the numbers either way.

**(ii) Fine-tuning carries normalization onto the GPU.** New in 1.8.0: when
`train` starts from a model, it re-applies that model's normalization stats on the
training device. This has only been tested on CPU.

```bash
varKoder train -m trained_pretrained -e 0 -z 2 --overwrite images ft_from_t1 2>&1 | tee ft_from_t1.log
python -c "import json; print(json.load(open('trained_pretrained/varkoder_config.json'))['normalize']); print(json.load(open('ft_from_t1/varkoder_config.json'))['normalize'])"
```

Expected: training runs on the GPU without a device-mismatch error (for example
"Expected all tensors to be on the same device"), and both printed `normalize`
values are the same non-null dict.

**(iii) A model trained on the GPU (mixed precision) reloads and queries on the
GPU.**

```bash
varKoder query -I -l ft_from_t1 images q_ft_gpu 2>&1 | tee q_ft_gpu.log
grep "will use" q_ft_gpu.log                               # expect cuda
head -3 q_ft_gpu/predictions.csv
```

**(iv) Default fine-tune from the published model on the GPU.** This downloads the
ViT-L model (~1.3 GB) and needs more GPU memory than resnet18; if it runs out of
memory, report that with the GPU model and memory.

```bash
varKoder train -e 0 -z 1 --overwrite images ft_default 2>&1 | tee ft_default.log
grep -E "Loading pretrained model from|GPU available" ft_default.log
python -c "import json; c=json.load(open('ft_default/varkoder_config.json')); print(c['architecture'], c['num_classes'], c['normalize'])"
```

Expected: it loads `brunoasm/vit_large_patch32_224.NCBI_SRA`, trains on the GPU,
and the saved config shows `vit_large_patch32_224` with `normalize` = `None` (the
published model was trained without normalization).

**(v) Multiple GPUs (only if `nvidia-smi` lists more than one).** Repeat (ii) with
all GPUs visible and confirm it runs (varKoder uses fastai's `to_parallel` when
more than one GPU is present). Otherwise write "single GPU, not tested".

## 2. Environment B: plain pip install (fastai 2.8)

`pip install varKoder` currently resolves to fastai 2.8. On 1.7.x that broke the
default model (fastai 2.8 cannot load its pickle); in 1.8.0 it should work because
the model now loads from safetensors.

```bash
conda deactivate
python3.11 -m venv "$WORK/venv-pip" && source "$WORK/venv-pip/bin/activate"
pip install torch torchvision          # default CUDA wheels
pip install -e "$REPO[test]"
python -c "import torch, fastai; print(torch.__version__, torch.cuda.is_available(), fastai.__version__)"
cd "$WORK"
varKoder query -I images q_default_pip 2>&1 | tee q_default_pip.log
grep "will use" q_default_pip.log                          # expect cuda
python -c "import pandas as pd; a=pd.read_csv('q_default_pip/predictions.csv'); b=pd.read_csv('q_default_gpu/predictions.csv'); print('identical predicted_labels:', (a['predicted_labels'].fillna('')==b['predicted_labels'].fillna('')).mean())"
deactivate
```

Expected: fastai 2.8.x, the query succeeds on the GPU with no pickle warning, and
`predicted_labels` match environment A's GPU run.

---

## Report template (`GPU_TEST_REPORT.md`)

```markdown
# varKoder 1.8.0 GPU validation report

- Commit tested:
- Date:
- Machine: GPU model(s) and memory, driver version, CUDA version (nvidia-smi), OS
- Environment A: Python / torch (+CUDA) / fastai / timm versions
- Environment B: Python / torch (+CUDA) / fastai / timm versions

## Results

| Step | Result (PASS / FAIL / SKIPPED) | Notes |
| --- | --- | --- |
| 1a unit tests (default / slow) | | counts |
| 1b integration suite | | list any failed commands |
| 1b training used GPU | | peak utilization / memory from gpu_usage.csv |
| 1b T1 config has normalize stats | | |
| 1c(i) default model GPU vs CPU | | rows, % identical labels, max prob diff |
| 1c(ii) fine-tune keeps normalization on GPU | | |
| 1c(iii) GPU-trained model queries on GPU | | |
| 1c(iv) default fine-tune from published model | | |
| 1c(v) multi-GPU | | |
| 2 pip install (fastai 2.8) default query | | |

## Failures

For each failure: the exact command, the full error message, and the relevant log
excerpt. Do not attempt fixes.

## TEST SUMMARY from suite.log

(paste)

## Anything else unexpected

(warnings, slowness, memory, etc.)
```
