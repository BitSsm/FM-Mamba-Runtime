"""One resumable epoch of a full-data experiment (one GPU process at a time).

Top-level imports stay lightweight for Windows DataLoader spawn.
"""
import _activate_runtime
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid


def learning_rate(config, step, batches):
    warmup = config["warmup_epochs"] * batches
    total = config["epochs"] * batches
    if step < warmup:
        return config["lr"] * (0.01 + 0.99*(step+1)/warmup)
    fraction = min(1., (step-warmup)/max(1, total-warmup-1))
    return config["min_lr"] + (config["lr"]-config["min_lr"])*0.5*(1+math.cos(math.pi*fraction))


def configure_torch(config):
    import torch
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def rng_state():
    import random
    import numpy as np
    import torch
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "python": random.getstate(),
            "numpy": [name, keys.tolist(), position, has_gauss, cached]}


def restore_rng(state):
    import random
    import numpy as np
    import torch
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))


def atomic_checkpoint(path, payload):
    import torch
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        os.replace(path, path.with_name("previous_checkpoint.pt"))
    os.replace(tmp, path)


def load_checkpoint(directory, recover=False):
    import torch
    if recover:
        errors = []
        for name in ("checkpoint.pt", "previous_checkpoint.pt"):
            candidate = directory / name
            if not candidate.exists():
                continue
            try:
                result = torch.load(candidate, map_location="cpu", weights_only=True)
                assert {"epoch", "next_batch", "step", "state_dict", "optimizer", "rng", "config_sha256", "dataset_fingerprint"} <= result.keys()
                if errors:
                    from coco_stream import write_json
                    for error in errors:
                        bad = Path(error["path"])
                        archived = directory / f"unreadable-{bad.stem}-{time.time_ns()}.pt"
                        os.replace(bad, archived)
                        error["preserved_as"] = str(archived)
                    write_json(directory / "checkpoint_recovery.json", {"errors": errors, "selected": str(candidate),
                               "step": result["step"], "time": datetime.now().astimezone().isoformat()})
                return result
            except Exception as exc:
                errors.append({"path": str(candidate), "error": repr(exc)})
        if errors:
            raise RuntimeError(f"No readable checkpoint: {errors}")
        return None
    path = directory / "checkpoint.pt"
    if not path.exists():
        path = directory / "previous_checkpoint.pt"
    return torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None


def evaluate(model, loader):
    import torch
    from torch.nn import functional as F
    from comparison_support import multilabel_metrics
    model.eval()
    outputs, targets, image_ids = [], [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for cpu, labels, ids in loader:
            x = cpu.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            outputs.append(model(x).float().cpu())
            targets.append(labels)
            image_ids.extend(ids.tolist())
    logits, targets = torch.cat(outputs), torch.cat(targets)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite validation logits")
    metrics = multilabel_metrics(logits, targets)
    metrics.update(loss=F.binary_cross_entropy_with_logits(logits, targets).item(),
                   wall_s=time.perf_counter()-start, samples=len(targets),
                   image_order_sha256=hashlib.sha256(json.dumps(image_ids).encode()).hexdigest())
    return metrics, logits


def run_worker(args):
    import gc
    import numpy as np
    import torch
    from torch.nn import functional as F
    from coco_stream import CocoImages, EpochBatches, make_loader, write_json
    from comparison_support import build_variant, make_optimizer, seed_all, step_diagnostics
    from run_comparison import core_gradient_norm, gradient_probe
    from resource_monitor import ResourceMonitor

    root = Path(args.run).resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    from holdout_training_plan import load_plan, check_resume_plan, planned_lr
    plan = load_plan(config)
    target_epochs = plan["target_epochs"] if plan else config["epochs"]
    if config["training_precision"] != "fp32" or not config["fm_recompute"]:
        raise ValueError("Full comparison requires matched FP32 and FM recomputation")
    configure_torch(config)
    directory = root / args.variant
    directory.mkdir(exist_ok=True)
    session = directory / "sessions" / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    session.mkdir(parents=True)
    metadata = json.loads((root / "data" / "dataset.json").read_text(encoding="utf-8"))
    fingerprint = metadata["fingerprint_sha256"]
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    train_data = CocoImages(root/"data", "train2017", config["resolution"], config["data_seed"], augment=True,
                            limit=config.get("test_train_limit"))
    val_data = CocoImages(root/"data", "val2017", config["resolution"], config["data_seed"],
                          limit=config.get("test_val_limit"))
    batches = math.ceil(len(train_data)/config["batch_size"])
    checkpoint = load_checkpoint(directory, recover=config.get("training_controller") == "holdout11")
    epoch, next_batch, step, best_map = 0, 0, 0, -1.
    accumulation = {"loss_sum": 0., "samples": 0, "train_wall_s": 0., "peak_allocated_mib": 0.}
    model, spec = build_variant(args.variant, config)
    spec.update(dataset_fingerprint=fingerprint, config_sha256=config_hash, batches_per_epoch=batches)
    spec_path = directory/"model_spec.json"
    if spec_path.exists():
        assert json.loads(spec_path.read_text(encoding="utf-8")) == spec, "Initialization/config changed"
    else:
        write_json(spec_path, spec)
    model.cuda()
    optimizer = make_optimizer(model, config)
    if checkpoint:
        assert checkpoint["variant"] == args.variant and checkpoint["config_sha256"] == config_hash
        assert checkpoint["dataset_fingerprint"] == fingerprint
        check_resume_plan(checkpoint.get("training_plan"), plan)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch, next_batch, step = checkpoint["epoch"], checkpoint["next_batch"], checkpoint["step"]
        best_map, accumulation = checkpoint["best_map"], checkpoint["accumulation"]
        saved_rng = checkpoint["rng"]
        del checkpoint
    else:
        seed_all(config["seed"]+99)
        saved_rng = rng_state()
    write_json(session/"resume_cursor.json", {"epoch": epoch, "next_batch": next_batch, "step": step,
               "config_sha256": config_hash, "dataset_fingerprint": fingerprint, "training_plan": plan})
    if epoch >= target_epochs:
        print(f"{args.variant}: already completed", flush=True)
        return 0
    sampler = EpochBatches(len(train_data), config["batch_size"], config["data_seed"], epoch, next_batch)
    order_hash = hashlib.sha256(sampler.indices().numpy().tobytes()).hexdigest()
    train_loader = make_loader(train_data, config, sampler)
    val_loader = make_loader(val_data, config)
    # Compile kernels outside training timings; eval does not consume model RNG.
    probe = torch.stack([val_data[i][0] for i in range(2)]).cuda().contiguous(memory_format=torch.channels_last)
    model.eval()
    with torch.inference_mode():
        model(probe)
    torch.cuda.synchronize()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    status = {"variant": args.variant, "pid": os.getpid(), "epoch_index": epoch, "epochs_completed": epoch,
              "next_batch": next_batch, "batches_per_epoch": batches, "step": step,
              "session": str(session), "status": "starting", "config_sha256": config_hash}

    def progress(phase, **extra):
        status.update(status=phase, step=step, epoch_index=epoch, next_batch=next_batch,
                      updated=datetime.now().astimezone().isoformat(), **extra)
        write_json(directory/"progress.json", status)

    def save():
        optimizer.zero_grad(set_to_none=True)
        atomic_checkpoint(directory/"checkpoint.pt", {"schema_version": 2, "variant": args.variant,
            "config": config, "config_sha256": config_hash, "dataset_fingerprint": fingerprint,
            "epoch": epoch, "next_batch": next_batch, "step": step, "best_map": best_map,
            "accumulation": accumulation, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "rng": rng_state(), "order_sha256": order_hash, "session": str(session), "training_plan": plan})

    with ResourceMonitor(session/"resources.csv", config["telemetry_interval_s"], include_children=True) as monitor, \
            (session/"train.csv").open("w", newline="", encoding="utf-8") as table:
        writer = None
        restore_rng(saved_rng)
        if step == 0 and not (directory/"initial_validation.json").exists():
            progress("initial_validation")
            monitor.set_phase("validation", step)
            metrics, _ = evaluate(model, val_loader)
            metrics.update(step=0, epoch=0)
            write_json(directory/"initial_validation.json", metrics)
            print(f"{args.variant}: initial val BCE={metrics['loss']:.5f}, mAP={metrics['map']:.4f}", flush=True)
        monitor.set_phase("diagnostics", step)
        model.eval()
        write_json(session/"diagnostics_start.json", step_diagnostics(model, probe))
        restore_rng(saved_rng)
        progress("training", epoch_order_sha256=order_hash)
        monitor.set_phase("train", step)
        iterator = iter(train_loader)
        steps_this_session = 0
        for batch_index in range(next_batch, batches):
            if (root/"PAUSE").exists():
                monitor.set_phase("checkpoint", step)
                save()
                progress("paused")
                break
            model.train()
            monitor.set_phase("train", step+1)
            start = time.perf_counter()
            batch_x, batch_y, image_ids = next(iterator)
            data_ms = (time.perf_counter()-start)*1000
            t0 = time.perf_counter()
            x = batch_x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = batch_y.cuda(non_blocking=True)
            torch.cuda.synchronize()
            h2d_ms = (time.perf_counter()-t0)*1000
            lr = planned_lr(config, plan, step, batches, learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            events[0].record()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            events[1].record()
            loss.backward()
            events[2].record()
            core_norm = core_gradient_norm(model, args.variant)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"], error_if_nonfinite=True)
            loss_value, norm_value, core_value = loss.item(), norm.item(), core_norm.item()
            if not all(np.isfinite(v) for v in (loss_value, norm_value, core_value)):
                raise FloatingPointError(f"Non-finite loss/gradient at {args.variant} {step+1}")
            optimizer.step()
            events[3].record()
            torch.cuda.synchronize()
            step_ms = (time.perf_counter()-start)*1000
            step, next_batch = step+1, batch_index+1
            row = {"step": step, "epoch": epoch+1, "batch": next_batch, "batch_size": len(image_ids),
                "image_ids": json.dumps(image_ids.tolist()), "loss": loss_value, "lr": lr,
                "grad_norm_before_clip": norm_value, "scan_core_grad_norm_before_clip": core_value,
                "loader_wait_ms": data_ms, "h2d_wall_ms": h2d_ms,
                "forward_event_span_ms": events[0].elapsed_time(events[1]),
                "backward_event_span_ms": events[1].elapsed_time(events[2]),
                "gradient_check_clip_optimizer_event_span_ms": events[2].elapsed_time(events[3]),
                "step_wall_ms": step_ms, "images_per_s": len(image_ids)*1000/step_ms,
                "allocated_mib": torch.cuda.memory_allocated()/2**20,
                "reserved_mib": torch.cuda.memory_reserved()/2**20,
                "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20}
            if writer is None:
                writer = csv.DictWriter(table, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            table.flush()
            accumulation["loss_sum"] += loss_value*len(image_ids)
            accumulation["samples"] += len(image_ids)
            accumulation["train_wall_s"] += step_ms/1000
            accumulation["peak_allocated_mib"] = max(accumulation["peak_allocated_mib"], row["peak_allocated_mib"])
            steps_this_session += 1
            if step == 1 or step % config["gradient_probe_every"] == 0:
                monitor.set_phase("diagnostics", step)
                write_json(session/f"gradients_{step}.json", gradient_probe(model, args.variant, step))
            del x, y, batch_x, batch_y, logits, loss, norm, core_norm
            if step == 1 or step % config["progress_every"] == 0:
                progress("training", last_loss=loss_value, last_lr=lr, last_step_ms=step_ms,
                         train_mean_loss=accumulation["loss_sum"]/accumulation["samples"],
                         observed_images_per_s=accumulation["samples"]/accumulation["train_wall_s"],
                         peak_allocated_mib=accumulation["peak_allocated_mib"])
                print(f"{args.variant} epoch {epoch+1}/{target_epochs} batch {next_batch}/{batches}: "
                      f"BCE={loss_value:.5f} lr={lr:.3g} {step_ms:.1f}ms peak={row['peak_allocated_mib']:.0f}MiB", flush=True)
            if step == 1 or step % config["checkpoint_every"] == 0 or next_batch == batches:
                monitor.set_phase("checkpoint", step)
                save()
            if args.max_steps and steps_this_session >= args.max_steps:
                monitor.set_phase("checkpoint", step)
                save()
                progress("preflight_stopped")
                break
        else:
            progress("validation")
            monitor.set_phase("validation", step)
            metrics, val_logits = evaluate(model, val_loader)
            metrics.update(step=step, epoch=epoch+1, train_loss=accumulation["loss_sum"]/accumulation["samples"],
                           train_samples=accumulation["samples"], train_wall_s=accumulation["train_wall_s"],
                           peak_allocated_mib=accumulation["peak_allocated_mib"], epoch_order_sha256=order_hash)
            assert accumulation["samples"] == len(train_data)
            epoch_dir = directory/"epochs"/f"{epoch+1:03d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            write_json(epoch_dir/"metrics.json", metrics)
            torch.save({"logits": val_logits, "image_ids": [r[0] for r in val_data.rows],
                        "category_ids": [c["id"] for c in metadata["categories"]]}, epoch_dir/"validation_logits.pt")
            monitor.set_phase("diagnostics", step)
            model.eval()
            write_json(epoch_dir/"step_diagnostics.json", step_diagnostics(model, probe))
            monitor.set_phase("checkpoint", step)
            if metrics["map"] > best_map:
                best_map = metrics["map"]
                temporary = directory/"best_weights.tmp"
                best_payload = {"state_dict": model.state_dict(), "variant": args.variant, "config": config,
                            "step": step, "epoch": epoch+1, "map": best_map,
                            "selection": "maximum full validation macro mAP", "dataset_fingerprint": fingerprint}
                with temporary.open("wb") as stream:
                    torch.save(best_payload, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                if config.get("training_controller") == "holdout11" and (directory/"best_weights.pt").exists():
                    os.replace(directory/"best_weights.pt", directory/"previous_best_weights.pt")
                os.replace(temporary, directory/"best_weights.pt")
            print(f"EPOCH COMPLETE {args.variant} {epoch+1}: train BCE={metrics['train_loss']:.5f}, "
                  f"val BCE={metrics['loss']:.5f}, mAP={metrics['map']:.4f}", flush=True)
            epoch, next_batch = epoch+1, 0
            accumulation = {"loss_sum": 0., "samples": 0, "train_wall_s": 0., "peak_allocated_mib": 0.}
            save()
            progress("completed" if epoch == target_epochs else "epoch_completed", epochs_completed=epoch,
                     best_map=best_map, last_validation={k: metrics[k] for k in
                         ("epoch", "step", "loss", "map", "micro_f1", "macro_f1", "samples", "wall_s")})
        monitor.set_phase("finished", step)
    write_json(session/"resource_summary.json", {phase: monitor.summary(phase)
               for phase in ("train", "validation", "diagnostics", "checkpoint")})
    del train_loader, val_loader, model, optimizer, probe
    gc.collect()
    return 75 if status["status"] == "paused" else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--max-steps", type=int, default=0, help="Only for independent preflight runs")
    args = parser.parse_args()
    try:
        return run_worker(args)
    except BaseException as exc:
        from coco_stream import write_json
        path = Path(args.run)/args.variant/"failure.json"
        write_json(path, {"error": repr(exc), "traceback": traceback.format_exc(),
                          "time": datetime.now().astimezone().isoformat(), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
