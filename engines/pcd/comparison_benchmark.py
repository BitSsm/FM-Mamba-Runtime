"""Repeated, randomized inference timing of the seven final pilot checkpoints."""
from contextlib import nullcontext
import csv
import gc
from pathlib import Path
import random
import time

import numpy as np
import torch

from comparison_support import VARIANTS, build_variant, json_write, read_image
from resource_monitor import ResourceMonitor


def measure(model, images, paths, precision, options):
    size, batch = images.shape[-1], len(images)
    x = images.cuda().contiguous(memory_format=torch.channels_last)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(options["iterations"])]
    rows = []
    with torch.inference_mode(), context:
        for _ in range(options["warmup"]):
            warm = model(x)
        if not torch.isfinite(warm).all():
            raise FloatingPointError("Non-finite inference output")
        del warm
        for start, end in pairs:
            start.record()
            end.record()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()/2**20
        torch.cuda.reset_peak_memory_stats()
        for index, (start, end) in enumerate(pairs):
            t0 = time.perf_counter()
            start.record()
            output = model(x)
            end.record()
            torch.cuda.synchronize()
            rows.append({"sample": index, "wall_ms": (time.perf_counter()-t0)*1000,
                         "cuda_event_span_ms": start.elapsed_time(end)})
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(options["throughput_iterations"]):
            output = model(x)
        torch.cuda.synchronize()
        throughput_wall = time.perf_counter()-t0
        peak = torch.cuda.max_memory_allocated()/2**20
        peak_reserved = torch.cuda.max_memory_reserved()/2**20
        if not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite inference output")
        end_to_end = []
        # This separate path includes JPEG decode, preprocessing, H2D and model.
        # Repeated inputs normally hit the OS file cache; not a cold-disk metric.
        for _ in range(5):
            t0 = time.perf_counter()
            cpu = torch.stack([read_image(p, size) for p in paths])
            gpu = cpu.cuda().contiguous(memory_format=torch.channels_last)
            result = model(gpu)
            torch.cuda.synchronize()
            end_to_end.append((time.perf_counter()-t0)*1000)
            del cpu, gpu, result
    return {"precision": precision, "resolution": size, "batch_size": batch,
        "warmup": options["warmup"], "samples": rows,
        "sustained_images_per_s": batch*options["throughput_iterations"]/throughput_wall,
        "sustained_wall_s": throughput_wall, "resident_allocated_mib": base,
        "peak_allocated_mib": peak, "peak_increment_mib": peak-base, "peak_reserved_mib": peak_reserved,
        "end_to_end_wall_ms_samples": end_to_end, "finite_output": True}


def profile_resources(model, images, precision, path, config):
    """Separate fixed-work profile: NVML polling can perturb request latency."""
    x = images.cuda().contiguous(memory_format=torch.channels_last)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    with ResourceMonitor(path, config["telemetry_interval_s"]) as monitor:
        monitor.set_phase("forward_profile")
        with torch.inference_mode(), context:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(60):
                output = model(x)
            torch.cuda.synchronize()
            wall = time.perf_counter()-start
        monitor.set_phase("complete")
    result = monitor.summary("forward_profile")
    result.update(forward_calls=60, measured_wall_s=wall,
                  scope="separate instrumented resource profile; not included in latency statistics")
    return result


def benchmark_all(config, dataset, root, directory_name="inference", resource_profile=True, checkpoint_name="checkpoint.pt", variants=None):
    variants = list(VARIANTS if variants is None else variants)
    if not variants or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS):
        raise ValueError("Require distinct supported benchmark variants")
    options = config["inference"]
    directory = root/directory_name
    directory.mkdir()
    inputs, paths = {}, {}
    source = dataset["splits"]["val2017"]["samples"]
    for size, batch in options["cases"]:
        paths[(size, batch)] = [Path(row["path"]) for row in source[:batch]]
        inputs[(size, batch)] = torch.stack([read_image(p, size) for p in paths[(size, batch)]])
    results, execution_order = [], []
    for round_index in range(options["rounds"]):
        order = VARIANTS.copy()
        random.Random(config["seed"]+300+round_index).shuffle(order)
        order = [v for v in order if v in variants]
        execution_order.append(order)
        for variant in order:
            model, _ = build_variant(variant, config)
            payload = torch.load(root/variant/checkpoint_name, map_location="cpu", weights_only=True, mmap=True)
            model.load_state_dict(payload["state_dict"], strict=True)
            del payload
            model.cuda().eval()
            cases = [(precision, size, batch) for precision in options["precisions"] for size, batch in options["cases"]]
            random.Random(config["seed"]+400+round_index).shuffle(cases)
            current = []
            for precision, size, batch in cases:
                gc.collect()
                torch.cuda.empty_cache()
                phase = f"{precision}_{size}_b{batch}"
                # No ResourceMonitor instance/thread or NVML query during timings.
                record = measure(model, inputs[(size, batch)], paths[(size, batch)], precision, options)
                record.update(variant=variant, round=round_index, phase=phase, telemetry_during_timing=False)
                if resource_profile and round_index == 0:
                    record["separate_resource_profile"] = profile_resources(
                        model, inputs[(size, batch)], precision, directory/f"resources_{variant}_{phase}.csv", config)
                current.append(record)
                print(f"Inference r{round_index+1} {variant} {precision} {size} b{batch}: "
                      f"{np.median([s['wall_ms'] for s in record['samples']]):.3f} ms median, "
                      f"{record['sustained_images_per_s']:.1f} img/s", flush=True)
            results.extend(current)
            json_write(directory/"rounds.json", results)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    aggregates, raw = [], []
    for variant in variants:
        for precision in options["precisions"]:
            for size, batch in options["cases"]:
                groups = [r for r in results if (r["variant"], r["precision"], r["resolution"], r["batch_size"]) == (variant, precision, size, batch)]
                samples = [s["wall_ms"] for g in groups for s in g["samples"]]
                event = [s["cuda_event_span_ms"] for g in groups for s in g["samples"]]
                for group in groups:
                    raw.extend({"variant": variant, "precision": precision, "resolution": size, "batch_size": batch,
                                "round": group["round"], **sample} for sample in group["samples"])
                e2e = [s for g in groups for s in g["end_to_end_wall_ms_samples"]]
                sustained = [g["sustained_images_per_s"] for g in groups]
                aggregates.append({"variant": variant, "precision": precision, "resolution": size, "batch_size": batch,
                    "rounds": len(groups), "samples": len(samples), "mean_ms": float(np.mean(samples)),
                    "p50_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
                    "p99_ms": float(np.percentile(samples, 99)), "std_ms": float(np.std(samples)),
                    "cuda_event_span_mean_ms": float(np.mean(event)),
                    "synchronized_images_per_s": batch*1000/float(np.mean(samples)),
                    "sustained_images_per_s": float(np.mean(sustained)),
                    "sustained_round_std_images_per_s": float(np.std(sustained)),
                    "peak_allocated_mib": max(g["peak_allocated_mib"] for g in groups),
                    "peak_reserved_mib": max(g["peak_reserved_mib"] for g in groups),
                    "peak_increment_mib": max(g["peak_increment_mib"] for g in groups),
                    "end_to_end_mean_ms": float(np.mean(e2e)), "end_to_end_p95_ms": float(np.percentile(e2e, 95)),
                    "round_p50_ms": [float(np.median([s["wall_ms"] for s in g["samples"]])) for g in groups]})
    for filename, rows in (("samples.csv", raw), ("summary.csv", aggregates)):
        with (directory/filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"status": "completed", "checkpoint_name": checkpoint_name, "execution_order": execution_order, "aggregates": aggregates,
        "scope": "Complete 80-class classifier; GPU-resident identical image tensors; compilation/loading excluded",
        "cuda_event_note": "Elapsed span includes stream idle/CPU launch gaps; not a sum of kernel runtimes",
        "precision_note": "FP32 is the primary matched recurrence comparison. BF16 autocast: FM scan still FP32, upstream scan BF16.",
        "e2e_scope": "JPEG decode+resize+normalize+H2D+model, five repeats/round; OS file cache is warm; telemetry disabled during timing",
        "telemetry_policy": "No resource sampler during latency/throughput/e2e measurement; resources profiled separately",
        "separate_resource_profile_in_this_pass": resource_profile,
        "statistics_note": "Empirical quantiles from short repeated runs, one GPU/seed; no statistical superiority claim"}
    json_write(directory/"summary.json", summary)
    return summary
