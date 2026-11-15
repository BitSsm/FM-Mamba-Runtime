"""Run a reproducible seven-model COCO multi-label pilot on the local GPU.

Usage: .\run.ps1 run_comparison.py --config configs/fm-seven-pilot.json
Outputs are always created in a new directory; no historical run is overwritten.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
import traceback
from datetime import datetime

import numpy as np
import torch
from torch.nn import functional as F

from comparison_support import (VARIANTS, batch_schedule, build_variant, coco_dataset,
    json_write, make_optimizer, multilabel_metrics, seed_all, step_diagnostics, tensor_hash)
from resource_monitor import ResourceMonitor


def command(args):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()


def evaluate(model, split, batch_size):
    model.eval()
    x, y = split
    outputs = []
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(x), batch_size):
            result = model(x[i:i+batch_size].cuda(non_blocking=True).contiguous(memory_format=torch.channels_last))
            outputs.append(result.float().cpu())
    logits = torch.cat(outputs)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite validation logits")
    metrics = multilabel_metrics(logits, y)
    metrics.update(loss=F.binary_cross_entropy_with_logits(logits, y).item(),
                   wall_s=time.perf_counter()-start, samples=len(x))
    return metrics


def core_gradient_norm(model, variant):
    keys = (".fm_core.",) if variant != "mamba2d" else (".x_proj.", ".dt_projT.", ".dt_projL.", ".AT_log", ".AL_log")
    if variant in ("vision_mamba_scan", "vmamba_scan", "two_d_mamba_scan"):
        keys = (".scan_core.",)
    if variant in ("vision_mamba", "vmamba", "two_d_mamba"):
        from models.external_mixers import VisionMambaMixer, SS2DMixer
        keys = tuple(name + "." for name, module in model.named_modules() if isinstance(module, (VisionMambaMixer, SS2DMixer)))
    parameters = [p for n, p in model.named_parameters() if any(k in n for k in keys) and p.grad is not None]
    return torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in parameters]))


def gradient_probe(model, variant, step):
    keys = (".fm_core.",) if variant != "mamba2d" else (".x_proj.", ".dt_projT.", ".dt_projL.", ".AT_log", ".AL_log")
    if variant in ("vision_mamba_scan", "vmamba_scan", "two_d_mamba_scan"):
        keys = (".scan_core.",)
    if variant in ("vision_mamba", "vmamba", "two_d_mamba"):
        from models.external_mixers import VisionMambaMixer, SS2DMixer
        keys = tuple(name + "." for name, module in model.named_modules() if isinstance(module, (VisionMambaMixer, SS2DMixer)))
    records = []
    for name, parameter in model.named_parameters():
        if not any(key in name for key in keys):
            continue
        grad = parameter.grad
        records.append({"parameter": name, "missing": grad is None,
                        "post_clip_grad_norm": grad.norm().item() if grad is not None else None,
                        "grad_finite": bool(torch.isfinite(grad).all()) if grad is not None else False})
    if not all(not r["missing"] and r["grad_finite"] for r in records):
        raise FloatingPointError("Missing or non-finite scan parameter gradient")
    return {"step": step, "parameters": records}


def train_one(variant, config, data, schedule, root):
    directory = root / variant
    directory.mkdir()
    model, spec = build_variant(variant, config)
    model.cuda()
    optimizer = make_optimizer(model, config)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    frozen_before = tensor_hash((k, v) for k, v in initial.items() if k.endswith((".A0", ".A1", ".A2")))
    x_train, y_train = data[0]
    x_val, _ = data[1]
    first_x = x_train[schedule[0]].cuda().contiguous(memory_format=torch.channels_last)
    first_y = y_train[schedule[0]].cuda()
    # Allocate kernels/optimizer state, then restore all initial weights and state.
    model.train()
    for _ in range(config["warmup_steps"]):
        optimizer.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(model(first_x), first_y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
        optimizer.step()
    torch.cuda.synchronize()
    model.load_state_dict(initial, strict=True)
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                value.zero_()
    optimizer.zero_grad(set_to_none=True)
    assert tensor_hash(model.state_dict().items()) == spec["state_dict_initial_sha256"]
    del initial, first_x, first_y
    gc.collect()
    torch.cuda.empty_cache()
    seed_all(config["seed"]+99)
    train_rows, evaluations, diagnostics, gradient_probes = [], [], [], []
    summary = {"status": "running", "model": spec, "config": config}
    json_write(directory/"summary.json", summary)
    best_loss = float("inf")
    probe = x_val[:min(2, config["batch_size"])].cuda().contiguous(memory_format=torch.channels_last)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    with ResourceMonitor(directory/"resources.csv", config["telemetry_interval_s"]) as monitor, \
            (directory/"train.csv").open("w", newline="", encoding="utf-8") as table, \
            (directory/"steps.jsonl").open("w", encoding="utf-8") as journal:
        writer = None
        def validation(step):
            monitor.set_phase("validation", step)
            metrics = evaluate(model, data[1], config["batch_size"])
            metrics["step"] = step
            evaluations.append(metrics)
            json_write(directory/"validation.json", evaluations)
            monitor.set_phase("diagnostics", step)
            diagnostics.append({"step": step, "modules": step_diagnostics(model, probe)})
            json_write(directory/"step_diagnostics.json", diagnostics)
            return metrics
        validation(0)
        for step, indices in enumerate(schedule, 1):
            model.train()
            monitor.set_phase("train", step)
            t_step = time.perf_counter()
            t_data = time.perf_counter()
            # Indexing returns pageable memory; pin explicitly so copy semantics
            # are identical for all variants. Disk decode is outside this timer.
            batch_x = x_train[indices].pin_memory()
            batch_y = y_train[indices].pin_memory()
            data_ms = (time.perf_counter()-t_data)*1000
            t_h2d = time.perf_counter()
            x = batch_x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = batch_y.cuda(non_blocking=True)
            torch.cuda.synchronize()
            h2d_ms = (time.perf_counter()-t_h2d)*1000
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            events[0].record()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            events[1].record()
            loss.backward()
            events[2].record()
            core_norm = core_gradient_norm(model, variant)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            optimizer.step()
            events[3].record()
            torch.cuda.synchronize()
            step_ms = (time.perf_counter()-t_step)*1000
            loss_value, norm_value, core_value = loss.item(), norm.item(), core_norm.item()
            if not all(np.isfinite(v) for v in (loss_value, norm_value, core_value)) or core_value <= 0:
                raise FloatingPointError(f"Invalid loss/gradient at {variant} step {step}")
            row = {"step": step, "sample_indices": json.dumps(indices.tolist()),
                "loss": loss_value, "lr": optimizer.param_groups[0]["lr"], "grad_norm_before_clip": norm_value,
                "scan_core_grad_norm_before_clip": core_value, "data_prepare_ms": data_ms, "h2d_wall_ms": h2d_ms,
                "forward_event_span_ms": events[0].elapsed_time(events[1]),
                "backward_event_span_ms": events[1].elapsed_time(events[2]),
                "gradient_check_clip_optimizer_event_span_ms": events[2].elapsed_time(events[3]),
                "step_wall_ms": step_ms, "images_per_s": config["batch_size"]*1000/step_ms,
                "allocated_mib": torch.cuda.memory_allocated()/2**20,
                "reserved_mib": torch.cuda.memory_reserved()/2**20,
                "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
                "loss_finite": True, "grad_finite": True}
            if writer is None:
                writer = csv.DictWriter(table, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            table.flush()
            journal.write(json.dumps(row, allow_nan=False)+"\n")
            journal.flush()
            train_rows.append(row)
            del logits, loss, x, y, batch_x, batch_y, core_norm, norm
            if step == 1 or step % config["eval_every"] == 0 or step == config["steps"]:
                monitor.set_phase("gradient_diagnostics", step)
                gradient_probes.append(gradient_probe(model, variant, step))
                json_write(directory/"gradient_probes.json", gradient_probes)
            if step % config["eval_every"] == 0 or step == config["steps"]:
                metrics = validation(step)
                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    monitor.set_phase("checkpoint", step)
                    torch.save({"state_dict": model.state_dict(), "step": step, "config": config,
                                "variant": variant, "selection": "minimum subset validation BCE"}, directory/"best_weights.pt")
                print(f"{variant} {step}/{config['steps']}: train BCE={loss_value:.5f}, val BCE={metrics['loss']:.5f}, "
                      f"subset mAP={metrics['map']:.4f}, peak={row['peak_allocated_mib']:.0f} MiB", flush=True)
        monitor.set_phase("checkpoint", config["steps"])
        optimizer.zero_grad(set_to_none=True)
        checkpoint_path = directory/"checkpoint.pt"
        torch.save({"schema_version": 1, "variant": variant, "config": config, "step": config["steps"],
                    "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate(), "sample_schedule": [i.tolist() for i in schedule]}, checkpoint_path)
        monitor.set_phase("reload_check")
        model.eval()
        with torch.inference_mode():
            expected = model(probe).cpu()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        # Perturb then strictly restore, so this checks the serialized data.
        with torch.no_grad():
            first_parameter = next(model.parameters())
            first_parameter[tuple(0 for _ in first_parameter.shape)].add_(1)
        model.load_state_dict(payload["state_dict"], strict=True)
        with torch.inference_mode():
            actual = model(probe).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        frozen_after = tensor_hash((k, v) for k, v in model.state_dict().items() if k.endswith((".A0", ".A1", ".A2")))
        assert frozen_before == frozen_after
        final_hash = tensor_hash(model.state_dict().items())
        assert final_hash != spec["state_dict_initial_sha256"]
        del payload
        monitor.set_phase("complete")
    resources = monitor.summary("train")
    timing = np.array([r["step_wall_ms"] for r in train_rows])
    summary.update(status="completed", steps=len(train_rows), images_seen=len(train_rows)*config["batch_size"],
        initial_validation=evaluations[0], final_validation=evaluations[-1],
        first_train_loss=train_rows[0]["loss"], final_train_loss=train_rows[-1]["loss"],
        mean_train_step_ms=float(timing.mean()), p50_train_step_ms=float(np.median(timing)),
        p95_train_step_ms=float(np.percentile(timing, 95)), timed_train_steps_s=float(timing.sum()/1000),
        train_images_per_s=config["batch_size"]*1000/float(timing.mean()),
        peak_train_allocated_mib=max(r["peak_allocated_mib"] for r in train_rows),
        peak_train_reserved_mib=max(r["peak_reserved_mib"] for r in train_rows), training_resources=resources,
        serialized_reload_max_abs_error=(actual-expected).abs().max().item(),
        frozen_continuous_A_unchanged=True if variant != "mamba2d" else None,
        final_state_sha256=final_hash, checkpoint_bytes=checkpoint_path.stat().st_size,
        gradient_diagnostics="All steps finite, scan-core gradient norm strictly positive",
        timing_scope="Step wall includes CPU batch selection/pinning, H2D, forward, loss, backward, gradient norm/clip, AdamW, synchronization; excludes CSV write, eval, probe, checkpoint and initial warmup")
    json_write(directory/"summary.json", summary)
    json_write(directory/"resource_summary.json", {"train": resources, "segments": monitor.segments, "errors": sorted(monitor.errors)})
    del model, optimizer, probe
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def source_snapshot(root):
    paths = set(Path("models").rglob("*.py")) | set(Path("models").rglob("*.cu")) | set(Path("models").rglob("*.cpp"))
    paths |= {p for p in Path(".").glob("*.py")} | {Path("run.ps1"), Path("configs/M2D-T-ImageNet.yaml"), Path("configs/fm-seven-pilot.json"), Path("requirements-experiments.txt")}
    hashes = {}
    for path in sorted(paths):
        target = root/"source_snapshot"/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root/"git-diff.patch").write_text(command(["git", "diff", "--binary"]), encoding="utf-8")
    (root/"packages.txt").write_text("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())), encoding="utf-8")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fm-seven-pilot.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--preflight", action="store_true", help="2 updates/model and a tiny inference check")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.preflight:
        config.update(steps=2, eval_every=2, train_samples=64, val_samples=64, warmup_steps=1)
        config["inference"].update(rounds=1, warmup=2, iterations=3, throughput_iterations=3, cases=[[224, 1]], precisions=["fp32"])
    if config["task"] != "coco_multilabel" or config["training_precision"] != "fp32":
        raise ValueError("This pilot runner implements COCO multi-label FP32 training")
    if not torch.cuda.is_available() or config["steps"] < 1 or config["train_samples"] < config["batch_size"]:
        raise ValueError("Require CUDA and positive run dimensions")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    root = Path(args.output_dir or f"outputs/fm-seven-{'preflight' if args.preflight else 'pilot'}-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    json_write(root/"config.json", config)
    manifest = {"status": "running", "schema_version": 1, "started": datetime.now().astimezone().isoformat(),
        "config": config, "environment": {"os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "tf32": False, "cudnn_deterministic": True, "cudnn_benchmark": False,
            "torch_compile_disabled": os.environ.get("TORCHDYNAMO_DISABLE"),
            "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,uuid", "--format=csv"])},
        "source_git_head": command(["git", "rev-parse", "HEAD"]), "git_status": command(["git", "status", "--short"]),
        "source_sha256": source_snapshot(root), "training": {}}
    json_write(root/"manifest.json", manifest)
    try:
        for verification in ("outputs/fm-matrix-verification.json", "outputs/fm-seven-legacy-regression/operator-verification.json", "outputs/coco2017-inspection.json"):
            if Path(verification).exists():
                shutil.copy2(verification, root/Path(verification).name)
        data, dataset = coco_dataset(config, root)
        manifest["dataset_fingerprint"] = dataset["fingerprint_sha256"]
        schedule = batch_schedule(config)
        json_write(root/"sample_schedule.json", [i.tolist() for i in schedule])
        order = VARIANTS.copy()
        random.Random(config["seed"]).shuffle(order)
        manifest["training_order"] = order
        json_write(root/"manifest.json", manifest)
        for variant in order:
            print(f"\nTraining {variant} ...", flush=True)
            manifest["active_variant"] = variant
            json_write(root/"manifest.json", manifest)
            manifest["training"][variant] = train_one(variant, config, data, schedule, root)
            json_write(root/"manifest.json", manifest)
        shared = [s["model"] for s in manifest["training"].values()]
        assert len({s["shared_outer_sha256"] for s in shared}) == 1
        assert len({s["matched_steps_C_D_sha256"] for s in shared}) == 1
        manifest["initialization_checks"] = {"shared_outer_weights_identical": True, "initial_steps_C_D_identical": True}
        from comparison_benchmark import benchmark_all
        manifest["inference"] = benchmark_all(config, dataset, root)
        from comparison_report import create_report
        manifest["status"] = "completed"
        manifest["finished"] = datetime.now().astimezone().isoformat()
        json_write(root/"manifest.json", manifest)
        create_report(root)
        print(f"\nCOMPLETED: {root}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        json_write(root/"manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
