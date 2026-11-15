# FM-Mamba Runtime

14组 COCO 80类多标签分类实验的运行版本。模型外壳、训练、评估和数据划分代码保存在仓库中；FM扫描、公式计算、调度与Graph管理通过预编译wheel提供。

## 环境与安装

当前版本：Windows x64、CPython 3.11、PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128、CUDA计算能力12.0。验证硬件为 RTX 5090 D。使用wheel安装，运行时直接加载预编译内核。

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

仓库当前为私有，`gh` 使用有仓库访问权限的账号。wheel版本与校验值随Release提供。

安装后，`.\run.ps1 tools/verify_runtime.py` 检查14组前向；`.\run.ps1 tools/short_check.py --data-root D:/datasets/coco2017` 执行PCD工况II的短程训练、续训、评估和测速。

## 实验组

|组名|结构与执行方式|
|---|---|
|`vim`、`vmamba`、`2dmamba`|前两阶段完整混合模块替换，后两阶段Transformer；状态维数依次为16、1、1|
|`m2d_2m2a`、`m2d_4m0a`|Mamba2D前两阶段并联外壳，后两阶段分别为Transformer、轻量串联外壳|
|`fm_2m2a_c1` / `c2` / `c3`|2M_FM 2A_T，通用融合；工况II叠加公式优化|
|`fm_4m0a_c1` / `c2` / `c3`|2M_FM 2A_FM，同上|
|`fm_pcd_c1` / `c2` / `c3`|2M_FM 2A_FM，三工况公式优化、形状调度、反向优化、完整状态保存、CUDA Graph及尾批Graph释放|

所有FM组采用就地输入，工况III的Δ上限为 `0.9 × ln(5) / 16`。主干深度为 `[3,3,9,3]`，通道为 `[64,128,320,512]`。完整组名与配置见 [configs/groups.json](configs/groups.json)。

## 数据与训练

固定划分：94,627张训练图、5,000张验证图、23,657张测试图。图像标签取实例类别并集，包含crowd类别，保留空标签图像。数据目录包含 `train2017`、`val2017`（也支持放在 `images/` 下）及 `annotations/instances_*.json`。

```powershell
.\run.ps1 prepare_data.py --data-root D:/datasets/coco2017 --output outputs/coco_index
.\run.ps1 experiments.py prepare --group fm_pcd_c2 --index outputs/coco_index --run outputs/c2
.\run.ps1 experiments.py train --group fm_pcd_c2 --run outputs/c2
.\run.ps1 experiments.py pause --group fm_pcd_c2 --run outputs/c2
.\run.ps1 experiments.py train --group fm_pcd_c2 --run outputs/c2 --resume
```

默认40轮、batch64、224×224、FP32、AdamW，训练seed为20261010，数据seed为42042。检查点保存权重、优化器、随机数状态和批次游标。`prepare --smoke` 使用99张训练图、16张验证图；`train --max-steps 1` 可检查保存与续训。正式训练可在测试集评估前通过 `extend --epochs 60 --lr 1e-6 --min-lr 1e-7` 追加训练段。

## 评估与测速

```powershell
.\run.ps1 experiments.py evaluate --group fm_pcd_c2 --run outputs/c2 --split val
.\run.ps1 experiments.py evaluate --group fm_pcd_c2 --run outputs/c2 --split test --index outputs/coco_index
.\run.ps1 experiments.py benchmark --group fm_pcd_c2 --run outputs/c2 --batch 1 --resolution 224 --trace
```

验证mAP用于选择最佳权重；测试评估由 `--split test` 启动。输出包括BCE、mAP、micro/macro F1、precision/recall、逐类指标、逐步训练耗时、资源采样、推理延迟、吞吐及CUDA trace。Graph用于PCD三组。

## Python调用与目录

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

`engines/` 为可读的模型外壳和训练代码，运行时从安装包加载FM二进制模块。`model_api.py` 提供模型构建、权重加载和Graph接口，`experiments.py` 提供完整流程。两类引擎在不同Python进程中运行。第三方许可与版本记录见 [THIRD_PARTY](THIRD_PARTY)，短程验证记录见 [VALIDATION.md](VALIDATION.md)。
