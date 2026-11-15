"""Shared-carrier initialization for three external 2M2A module baselines."""
import json

import torch

from local_support import build_classifier
from models.external_mixers import replace_first_two_stages
from models.kernels.external_scan import UPSTREAM


ARCHITECTURE_VERSION = "external_full_mixer_2m2a_20261024"


def build_external_variant(variant, config):
    from comparison_support import seed_all, tensor_hash
    if config.get("external_mixer_scope", "full_mixer") != "full_mixer":
        raise ValueError("This adapter replaces complete mixing modules")
    seed_all(config["seed"])
    model = build_classifier(config["model_config"])
    if config["task"] != "coco_multilabel":
        raise ValueError("External comparison uses the existing COCO multilabel protocol")
    model.head.fc2 = torch.nn.Linear(model.head.fc2.in_features, 80)
    model.n_classes = 80
    model.train_loss = torch.nn.BCEWithLogitsLoss()
    model.val_loss = torch.nn.BCEWithLogitsLoss()
    from models.mamba2d import M2DBlock
    paths = [name for name, module in model.named_modules() if isinstance(module, M2DBlock)]
    prefixes = tuple(name + "." for name in paths)
    common_keys = [name for name in model.state_dict() if not name.startswith(prefixes)]
    common = tensor_hash((name, model.state_dict()[name]) for name in common_keys)
    # Build the unmodified carrier first so its stem, FFNs, downsampling, final
    # two attention stages and head are bitwise identical to baseline at init.
    paths = replace_first_two_stages(model, variant)
    assert common == tensor_hash((name, model.state_dict()[name]) for name in common_keys)
    upstream_name = "vim" if variant == "vision_mamba" else variant
    upstream = json.loads((UPSTREAM / upstream_name / "upstream.json").read_text(encoding="utf-8"))
    spec = {
        "variant": variant, "architecture_version": ARCHITECTURE_VERSION,
        "comparison_scope": "complete mixing modules in stages 1/2 of the fixed Mamba2D-T carrier",
        "condition": None, "input_form": "upstream", "continuous_A": "learnable",
        "stage_token_mixers": list(model.backbone.token_mixer), "replaced_mixer_paths": paths,
        "replaced_mixers": len(paths), "stage_depths": [3, 3, 9, 3], "stage_widths": [64, 128, 320, 512],
        "d_state": 16 if variant == "vision_mamba" else 1,
        "inner_expansion": 2 if variant == "vision_mamba" else 1,
        "scan_directions_per_block": 2 if variant == "vision_mamba" else 4,
        "operator_recipe": "BiMamba-v2, div2, causal DWConv1d4, SiLU gate" if variant == "vision_mamba"
                           else "v05_noz, bias-free DWConv2d3, SiLU, four-direction sum, LayerNorm",
        "upstream_repository": upstream["repository"], "upstream_commit": upstream["commit"],
        "recurrence_training_dtype": "float32",
        "runtime_note": "native CUDA selective scan; PyTorch convolution/projections/cross-scan; no Triton or fused Vim inner op",
        "bf16_note": "2DMamba inputs promoted to FP32; VMamba/Vim CUDA accumulates scan in FP32",
        "state_storage": "upstream scan chunk states; native CUDA backward",
        "cuda_correctness_patch": "atomic B/C gradient accumulation across channel blocks" if variant == "two_d_mamba" else None,
        "shared_outer_sha256": common, "shared_carrier_tensor_count": len(common_keys),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_dict_initial_sha256": tensor_hash(model.state_dict().items()),
        "mixer_initial_sha256": tensor_hash((n, t) for n, t in model.state_dict().items() if n.startswith(prefixes)),
    }
    return model, spec
