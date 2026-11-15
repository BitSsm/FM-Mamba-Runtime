"""Shared initialization, diagnostic data, and metadata for the seven-way pilot."""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from local_support import build_classifier
from models.fm_mamba import FMSelective2D, replace_mamba2d_mixers

VARIANTS = ["mamba2d"] + [f"fm_c{c}_{f}" for c in (1, 2, 3) for f in ("local", "multi")]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_variant(variant, config):
    """Copy common outer weights, step projections and C/D exactly from baseline.

    Local B0 starts from (baseline BT+BL)/sqrt(2). Revised multi W_B1/W_B2
    contain the exact legacy B1/B2 seeded scalar values, transposed into linear
    projection weights. This preserves seed/range/values while changing the
    input dependence required by the two manuscript revisions.
    """
    if variant in ("vision_mamba_scan", "vmamba_scan", "two_d_mamba_scan"):
        from scan_comparison_support import build_scan_variant
        return build_scan_variant(variant, config)
    if variant in ("vision_mamba", "vmamba", "two_d_mamba"):
        from external_comparison_support import build_external_variant
        return build_external_variant(variant, config)
    from models.mamba2d import M2DBlock
    if variant not in VARIANTS:
        raise ValueError(variant)
    seed_all(config["seed"])
    model = build_classifier(config["model_config"])
    if config.get("task") == "coco_multilabel":
        model.head.fc2 = torch.nn.Linear(model.head.fc2.in_features, 80)
        model.n_classes = 80
        model.train_loss = torch.nn.BCEWithLogitsLoss()
        model.val_loss = torch.nn.BCEWithLogitsLoss()
    cores = [(name, module) for name, module in model.named_modules() if isinstance(module, M2DBlock)]
    core_keys = ("x_proj.", "dt_projT.", "dt_projL.", "AT_log", "AL_log", "D")
    excluded = tuple(name + "." + key for name, _ in cores for key in core_keys)
    outer_keys = [k for k in model.state_dict() if not k.startswith(excluded)]
    shared_hash = tensor_hash((k, model.state_dict()[k]) for k in outer_keys)
    dynamics = {name: {k: v.detach().clone() for k, v in module.state_dict().items()}
                for name, module in cores}
    spec = {"variant": variant, "condition": None, "input_form": "original", "continuous_A": "learnable",
            "recurrence_training_dtype": "float32", "state_storage": "upstream partial recomputation"}
    serial_paths = [name for name, module in cores
                    if getattr(module, "mixer_layout", None) == "vmamba_serial"]
    if serial_paths:
        spec.update(architecture_version="hybrid_vmamba_serial_20261016",
                    stage_token_mixers=list(model.backbone.token_mixer),
                    serial_mixer_paths=serial_paths,
                    parallel_mixer_paths=[name for name, _ in cores if name not in serial_paths],
                    serial_shell="Linear(no bias) -> DWConv3(no bias) -> SiLU -> scan -> LayerNorm -> Linear(no bias)",
                    direction_scope="Flip scan input/output only; convolution runs in original spatial orientation")
    if variant != "mamba2d":
        _, c, form = variant.split("_")
        condition = int(c[1:])
        replace_mamba2d_mixers(model, condition=condition, input_form=form,
                              stability_margin=config["stability_margin"], recompute=config.get("fm_recompute", False))
        with torch.no_grad():
            for index, (name, _) in enumerate(cores):
                fm = model.get_submodule(name).fm_core
                old = dynamics[name]
                r, n = fm.dt_rank, fm.d_state
                p = old["x_proj.weight"]
                if form == "local":
                    fm.x_proj.weight.copy_(torch.cat((p[:2*r], (p[2*r:2*r+n] + p[2*r+n:2*r+2*n]) / math.sqrt(2), p[-n:])))
                else:
                    generator = torch.Generator().manual_seed(config["seed"] + 1000 + index)
                    b1 = torch.empty(fm.d_model, n).uniform_(-n**-0.5, n**-0.5, generator=generator)
                    b2 = torch.empty(fm.d_model, n).uniform_(-n**-0.5, n**-0.5, generator=generator)
                    fm.x_proj.weight.copy_(torch.cat((p[:2*r], b1.T, b2.T, p[-n:])))
                for target, source in ((fm.dt_proj1, "dt_projT"), (fm.dt_proj2, "dt_projL")):
                    target.weight.copy_(old[source + ".weight"])
                    target.bias.copy_(old[source + ".bias"])
                fm.D.copy_(old["D"])
        assert shared_hash == tensor_hash((k, model.state_dict()[k]) for k in outer_keys)
        spec.update(condition=condition, input_form=form, continuous_A="frozen",
                    equation=42 if form == "local" else 20, input_mode="first_order",
                    formula_version="draft_20261007_red" if form == "local" else "revision2_20261012_dynamic_B_corrected_drive",
                    state_storage="all FP32 states saved for backward",
                    step_cap=(1-config["stability_margin"]) * math.log(5) / 16 if condition == 3 else None)
        if form == "multi":
            spec.update(drive="B1(x)*delta2*(x-x_top)+B2(x)*delta1*(x-x_left)",
                        B_parameterization="bias-free input-dependent projections, shared across feature channels",
                        drive_initialization="transpose legacy B1/B2 values; unchanged seed+1000+core_index and uniform(-1/sqrt(N),1/sqrt(N))")
        if config.get("fm_recompute", False):
            spec["state_storage"] = "scan states recomputed in backward; projections saved"
    # Canonical names let the initialization check span both parameter layouts.
    matched = []
    for name, _ in cores:
        module = model.get_submodule(name)
        core = module if variant == "mamba2d" else module.fm_core
        r, n = core.dt_rank, core.d_state
        matched += [(name + ".steps_x", core.x_proj.weight[:2*r]),
                    (name + ".C_x", core.x_proj.weight[-n:]), (name + ".D", core.D)]
        for index in (1, 2):
            layer = getattr(core, ("dt_projT" if index == 1 else "dt_projL") if variant == "mamba2d" else f"dt_proj{index}")
            matched += [(f"{name}.dt{index}.weight", layer.weight), (f"{name}.dt{index}.bias", layer.bias)]
    spec.update(shared_outer_sha256=shared_hash, matched_steps_C_D_sha256=tensor_hash(matched),
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                state_dict_initial_sha256=tensor_hash(model.state_dict().items()), replaced_mixers=len(cores))
    if "execution_backends" in config:
        execution_backend = config["execution_backends"][variant]
        if variant == "mamba2d":
            assert execution_backend == "original", "Original Mamba2D remains unoptimized"
        else:
            for module in model.modules():
                if isinstance(module, FMSelective2D):
                    module.set_execution_backend(execution_backend)
        spec["execution_backend"] = execution_backend
    if config.get("fm_runtime_profile") and variant.startswith("fm_"):
        from models.fm_runtime import apply_current_fm_optimizations
        options = config["fm_runtime_profile"]
        if isinstance(options, str):
            profile_path = Path(__file__).resolve().parent / options
            options = json.loads(profile_path.read_text(encoding="utf-8"))
        if options.get("profile") != "current_20261031":
            raise ValueError("Unknown FM runtime profile")
        spec["runtime_profile"] = apply_current_fm_optimizations(model,
            state_policy=options.get("state_policy", "preserve"), specialize=options.get("specialize", True))
        spec["execution_backend"] = "current_20261031"
    return model, spec


def synthetic_dataset(config):
    """Small, learnable colored stripe task; no ImageNet/generalization claims."""
    size, classes = config["resolution"], config["synthetic_classes"]
    axis = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    prototype = []
    for label in range(classes):
        pattern = torch.sin((label // 3 + 1) * math.pi * (xx if label % 2 else yy))
        base = torch.zeros(3, size, size)
        base[label % 3] = 0.8 + 0.45 * pattern
        base[(label + 1) % 3] = 0.3 * pattern
        prototype.append(base)
    prototype = torch.stack(prototype)
    splits = []
    for offset, count in ((0, config["train_samples"]), (1, config["val_samples"])):
        g = torch.Generator().manual_seed(config["data_seed"] + offset)
        labels = torch.arange(count) % classes
        labels = labels[torch.randperm(count, generator=g)]
        data = prototype[labels] + 0.18 * torch.randn(count, 3, size, size, generator=g)
        data += torch.randn(count, 3, 1, 1, generator=g) * 0.06
        splits.append((data.contiguous().pin_memory(), labels.pin_memory()))
    fingerprint = tensor_hash((name, tensor) for name, tensor in zip(
        ("train_x", "train_y", "val_x", "val_y"), (*splits[0], *splits[1])))
    return splits, {"type": "structured_synthetic", "fingerprint_sha256": fingerprint,
        "train_samples": config["train_samples"], "val_samples": config["val_samples"],
        "classes_in_data": classes, "classifier_outputs": 1000, "accuracy_scope": "diagnostic synthetic task only"}


def batch_schedule(config):
    g = torch.Generator().manual_seed(config["data_seed"] + 50)
    batches = []
    while len(batches) < config["steps"]:
        order = torch.randperm(config["train_samples"], generator=g)
        batches.extend(list(order[:len(order)//config["batch_size"]*config["batch_size"]].split(config["batch_size"])))
    return batches[:config["steps"]]


def make_optimizer(model, config):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if getattr(p, "_no_weight_decay", False) or p.ndim < 2 else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": config["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.}],
                             lr=config["lr"], betas=(0.9, 0.999), eps=1e-8, foreach=True)


def step_diagnostics(model, x):
    """Explicit probe outside timed steps; frozen-coefficient checks, not a proof."""
    records, handles = [], []
    def hook(name):
        def record(module, args):
            d1, d2, *_ = module.project(args[0], raw_steps=True)
            raw = torch.stack((d1, d2))
            effective = raw.clamp_max(module.step_cap) if module.step_cap is not None else raw
            r = module.A0 / (module.A1 * module.A2)
            # Positive dt plus this criterion is the draft's frozen scalar check.
            u = torch.exp(effective.unsqueeze(-1) * module.A1)
            criterion = (1 + r * (1-u)).abs()
            records.append({"module": name, "condition": module.condition, "input_form": module.input_form,
                "raw_min": raw.min().item(), "raw_max": raw.max().item(), "effective_max": effective.max().item(),
                "clipped_fraction": (raw > module.step_cap).float().mean().item() if module.step_cap else 0.,
                "frozen_criterion_max": criterion.max().item(),
                "frozen_violation_count": (criterion >= 1).sum().item(), "scope": "fixed diagnostic minibatch"})
        return record
    try:
        for name, module in model.named_modules():
            if isinstance(module, FMSelective2D):
                handles.append(module.register_forward_pre_hook(hook(name)))
        if handles:
            with torch.inference_mode():
                model(x)
    finally:
        for handle in handles:
            handle.remove()
    return records


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_image(path, size):
    from PIL import Image
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


def coco_dataset(config, directory):
    """Cover all categories first, then fill with seeded random images.

    This is a coverage-selected diagnostic subset, not a representative/full
    COCO evaluation. Full images are resized (no crop); all annotated categories,
    including crowd annotations, are positive image-level labels.
    """
    import gc
    from collections import defaultdict
    from inspect_coco import sha256
    root = Path(config["data_root"])
    splits, manifests = [], {}
    for offset, (split, count) in enumerate((("train2017", config["train_samples"]), ("val2017", config["val_samples"]))):
        annotation = root / "annotations" / f"instances_{split}.json"
        with annotation.open(encoding="utf-8") as stream:
            data = json.load(stream)
        categories = sorted(data["categories"], key=lambda c: c["id"])
        if offset == 0:
            category_map = categories
        else:
            assert categories == category_map
        cat_index = {c["id"]: i for i, c in enumerate(categories)}
        images = {i["id"]: i for i in data["images"]}
        labels = defaultdict(set)
        for obj in data["annotations"]:
            labels[obj["image_id"]].add(cat_index[obj["category_id"]])
        ordered = sorted(images)
        random.Random(config["data_seed"]+offset).shuffle(ordered)
        candidates = {i: [] for i in range(80)}
        for image_id in ordered:
            for c in labels[image_id]:
                candidates[c].append(image_id)
        selected, covered = [], set()
        for c in sorted(candidates, key=lambda c: (len(candidates[c]), c)):
            if c not in covered:
                image_id = candidates[c][0]
                selected.append(image_id)
                covered.update(labels[image_id])
        coverage_count = len(selected)
        if count < coverage_count:
            raise ValueError(f"Need at least {coverage_count} images for category coverage in {split}")
        used = set(selected)
        selected.extend(i for i in ordered if i not in used)
        selected = selected[:count]
        random.Random(config["data_seed"]+100+offset).shuffle(selected)
        x = torch.empty(count, 3, config["resolution"], config["resolution"], pin_memory=True)
        y = torch.zeros(count, 80, pin_memory=True)
        records = []
        for index, image_id in enumerate(selected):
            path = root / "images" / split / images[image_id]["file_name"]
            x[index].copy_(read_image(path, config["resolution"]))
            y[index, sorted(labels[image_id])] = 1
            records.append({"image_id": image_id, "path": str(path.resolve()), "image_sha256": sha256(path),
                            "positive_indices": sorted(labels[image_id])})
        assert (y.sum(0) > 0).all()
        manifests[split] = {"annotation_sha256": sha256(annotation), "samples": records,
            "category_coverage_prefix_count": coverage_count, "positives_per_class": y.sum(0).int().tolist(),
            "preprocessed_tensors_sha256": tensor_hash((("x", x), ("y", y)))}
        splits.append((x, y))
        print(f"Loaded {split}: {count} images, all 80 categories, {int(y.sum())} positive image labels", flush=True)
        del data, images, labels, candidates
        gc.collect()
    train_ids = {r["image_id"] for r in manifests["train2017"]["samples"]}
    val_ids = {r["image_id"] for r in manifests["val2017"]["samples"]}
    assert not train_ids & val_ids
    manifest = {"type": "coco_image_level_multilabel", "root": str(root), "categories": category_map,
        "preprocessing": "PIL RGB, full-image bicubic resize to square, ImageNet mean/std; no crop/augmentation",
        "targets": "union of annotated category IDs per image, including crowd; box coordinates unused",
        "sampling": "rare-category coverage first, seeded random fill; same fixed subset/order for all seven",
        "io_scope": "decode/preprocess once into pinned CPU cache before timed training; H2D timed separately",
        "accuracy_scope": "small coverage-selected subset; neither official COCO detection AP nor ImageNet accuracy",
        "train_samples": config["train_samples"], "val_samples": config["val_samples"], "splits": manifests}
    manifest["fingerprint_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    json_write(Path(directory)/"dataset_manifest.json", manifest)
    return splits, manifest


def multilabel_metrics(logits, targets):
    scores, truth = logits.double().cpu().numpy(), targets.cpu().numpy().astype(bool)
    prediction = scores >= 0  # sigmoid(logit) >= 0.5
    tp = (prediction & truth).sum(0)
    fp = (prediction & ~truth).sum(0)
    fn = (~prediction & truth).sum(0)
    def ratio(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    precision, recall = ratio(tp, tp+fp), ratio(tp, tp+fn)
    f1 = ratio(2*tp, 2*tp+fp+fn)
    aps = []
    for c in range(truth.shape[1]):
        positive = truth[:, c].sum()
        if not positive:
            aps.append(None)
            continue
        order = np.argsort(-scores[:, c], kind="stable")
        sorted_scores, sorted_truth = scores[order, c], truth[order, c]
        ends = np.r_[np.where(np.diff(sorted_scores) != 0)[0], len(order)-1]
        cumulative = np.cumsum(sorted_truth)[ends]
        recalls = cumulative/positive
        precisions = cumulative/(ends+1)
        aps.append(float(np.sum(np.diff(np.r_[0, recalls])*precisions)))
    return {"map": float(np.mean([a for a in aps if a is not None])), "ap_evaluable_classes": sum(a is not None for a in aps),
        "micro_precision": float(tp.sum()/max(1, (tp+fp).sum())),
        "micro_recall": float(tp.sum()/max(1, (tp+fn).sum())),
        "micro_f1": float(2*tp.sum()/max(1, (2*tp+fp+fn).sum())), "macro_f1": float(f1.mean()),
        "threshold": 0.5, "per_class_ap": aps, "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(), "per_class_positive_count": truth.sum(0).tolist()}
