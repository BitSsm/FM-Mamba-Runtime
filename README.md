# FM-Mamba Runtime

Runtime distribution for 14 COCO 80-class multilabel classification experiments. The repository contains readable model wrappers, training, evaluation, and data-split code. FM scans, coefficient calculations, scheduling, and CUDA Graph management are supplied in a precompiled wheel.

## Environment and installation

Supported environment: Windows x64, CPython 3.11, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, and CUDA compute capability 12.0. Validation hardware: RTX 5090 D. Install the wheel to load precompiled kernels at runtime.

```powershell
gh repo clone BitSsm/FM-Mamba-Runtime
cd FM-Mamba-Runtime
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt -c requirements-windows-lock.txt
gh release download v0.1.0 --repo BitSsm/FM-Mamba-Runtime --pattern "*.whl" --pattern SHA256SUMS.txt --dir dist
.\.venv\Scripts\python.exe -m pip install (Get-ChildItem dist/*.whl).FullName
.\run.ps1 experiments.py smoke --group fm_pcd_c2
```

Wheel versions and checksums are included in each release.

After installation, `.\run.ps1 tools/verify_runtime.py` checks forward execution for all 14 groups. `.\run.ps1 tools/short_check.py --data-root D:/datasets/coco2017` runs a short training, resume, evaluation, and benchmark cycle for PCD condition II.

## Experiment groups

|Group IDs|Architecture and execution|
|---|---|
|`vim`, `vmamba`, `2dmamba`|Full mixer replacement in the first two stages; Transformer attention in the last two. State dimensions: 16, 1, and 1, respectively.|
|`m2d_2m2a`, `m2d_4m0a`|Mamba2D with parallel local branches in the first two stages; Transformer attention or lightweight serial wrappers in the last two, respectively.|
|`fm_2m2a_c1` / `c2` / `c3`|2M_FM 2A_T with generic fusion; condition II also uses its formula optimization.|
|`fm_4m0a_c1` / `c2` / `c3`|2M_FM 2A_FM with generic fusion; condition II also uses its formula optimization.|
|`fm_pcd_c1` / `c2` / `c3`|2M_FM 2A_FM with formula optimizations for all three conditions, shape-based scheduling, backward optimizations, full state retention, CUDA Graph, and Graph release before the final partial batch.|

All FM groups use local input. The condition III step-size cap is `0.9 * ln(5) / 16`. Backbone depths are `[3,3,9,3]`, with channel widths `[64,128,320,512]`. Full group IDs and configuration paths are listed in [configs/groups.json](configs/groups.json).

## Data and training

The fixed split contains 94,627 training, 5,000 validation, and 23,657 test images. Image labels are the union of instance categories, including crowd annotations; images with empty labels are retained. The data directory contains `train2017`, `val2017` (also supported under `images/`), and `annotations/instances_*.json`.

```powershell
.\run.ps1 prepare_data.py --data-root D:/datasets/coco2017 --output outputs/coco_index
.\run.ps1 experiments.py prepare --group fm_pcd_c2 --index outputs/coco_index --run outputs/c2
.\run.ps1 experiments.py train --group fm_pcd_c2 --run outputs/c2
.\run.ps1 experiments.py pause --group fm_pcd_c2 --run outputs/c2
.\run.ps1 experiments.py train --group fm_pcd_c2 --run outputs/c2 --resume
```

Defaults: 40 epochs, batch size 64, 224 x 224 inputs, FP32, and AdamW. The training seed is 20261010; the data seed is 42042. Checkpoints store weights, optimizer state, RNG state, and the batch cursor. `prepare --smoke` selects 99 training and 16 validation images; `train --max-steps 1` supports a save-and-resume check. Before test evaluation, extend training with `extend --epochs 60 --lr 1e-6 --min-lr 1e-7`.

## Evaluation and benchmarking

```powershell
.\run.ps1 experiments.py evaluate --group fm_pcd_c2 --run outputs/c2 --split val
.\run.ps1 experiments.py evaluate --group fm_pcd_c2 --run outputs/c2 --split test --index outputs/coco_index
.\run.ps1 experiments.py benchmark --group fm_pcd_c2 --run outputs/c2 --batch 1 --resolution 224 --trace
```

Validation mAP selects the best weights. `--split test` starts test evaluation. Outputs include BCE, mAP, micro/macro F1, precision/recall, per-class metrics, per-step training times, resource samples, inference latency, throughput, and CUDA traces. CUDA Graph is enabled for the three PCD groups.

## Python API and repository layout

```python
import torch
from model_api import build_model, inference_graph

model, metadata = build_model('fm_pcd_c2', checkpoint='best_weights.pt')
model.eval()
with torch.inference_mode():
    x = torch.randn(1, 3, 224, 224, device='cuda')
    logits = model(x)
    graph = inference_graph(model, x)
    logits = graph.owned(x)
```

`engines/` contains readable model wrappers and training code that load FM binary modules from the installed package. `model_api.py` provides model construction, weight loading, and Graph interfaces; `experiments.py` provides the complete workflow. Run the reference and PCD engines in separate Python processes. See [THIRD_PARTY](THIRD_PARTY) for licenses and revision records, and [VALIDATION.md](VALIDATION.md) for short-run validation results.
