"""Audit historical pilot outputs and render plots and an English run report."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from comparison_support import VARIANTS, json_write, multilabel_metrics

LABELS = {"mamba2d": "Mamba2D", **{f"fm_c{c}_{f}": f"FM C{c} {f}" for c in (1, 2, 3) for f in ("local", "multi")}}
SHORT = ["M2D", "C1 L", "C1 M", "C2 L", "C2 M", "C3 L", "C3 M"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(root, manifest):
    import torch
    config = manifest["config"]
    checks = {}
    expected_eval = [0]+list(range(config["eval_every"], config["steps"]+1, config["eval_every"]))
    if expected_eval[-1] != config["steps"]:
        expected_eval.append(config["steps"])
    schedules = []
    for variant in VARIANTS:
        directory = root/variant
        summary = read_json(directory/"summary.json")
        with (directory/"train.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        journal = [json.loads(s) for s in (directory/"steps.jsonl").read_text(encoding="utf-8").splitlines()]
        validation = read_json(directory/"validation.json")
        probes = read_json(directory/"step_diagnostics.json")
        gradients = read_json(directory/"gradient_probes.json")
        assert all(not p["missing"] and p["grad_finite"] for g in gradients for p in g["parameters"])
        with (directory/"resources.csv").open(encoding="utf-8") as stream:
            resources = list(csv.DictReader(stream))
        assert summary["status"] == "completed"
        assert len(rows) == len(journal) == config["steps"]
        assert [int(r["step"]) for r in rows] == list(range(1, config["steps"]+1))
        assert [v["step"] for v in validation] == expected_eval
        for key in ("loss", "step_wall_ms", "grad_norm_before_clip", "scan_core_grad_norm_before_clip", "peak_allocated_mib"):
            assert all(np.isfinite(float(r[key])) for r in rows)
        assert all(float(r["scan_core_grad_norm_before_clip"]) > 0 for r in rows)
        assert summary["serialized_reload_max_abs_error"] == 0
        assert any(r["phase"] == "train" for r in resources)
        assert (directory/"checkpoint.pt").stat().st_size > 0
        assert (directory/"best_weights.pt").stat().st_size > 0
        assert all(v["ap_evaluable_classes"] == 80 for v in validation)
        if variant != "mamba2d":
            assert summary["frozen_continuous_A_unchanged"]
            assert all(m["frozen_violation_count"] == 0 for p in probes for m in p["modules"])
        schedules.append([r["sample_indices"] for r in rows])
        checks[variant] = {"step_csv_jsonl_complete": True, "finite_loss_and_gradients": True,
            "checkpoint_strict_reload_exact": True, "validation_complete": True, "telemetry_present": True,
            "frozen_coefficient_probes_pass": True if variant != "mamba2d" else "not applicable"}
    assert all(schedule == schedules[0] for schedule in schedules)
    assert len(manifest["training"]) == 7
    specs = [s["model"] for s in manifest["training"].values()]
    assert len({s["shared_outer_sha256"] for s in specs}) == 1
    assert len({s["matched_steps_C_D_sha256"] for s in specs}) == 1
    inference_directory = manifest.get("primary_inference_directory", "inference")
    inference = read_json(root/inference_directory/"summary.json")
    expected_count = 7*len(config["inference"]["cases"])*len(config["inference"]["precisions"])
    assert len(inference["aggregates"]) == expected_count
    assert all(row["samples"] == config["inference"]["rounds"]*config["inference"]["iterations"] for row in inference["aggregates"])
    # Hand-solvable rankings guard the scientific metric definitions, including ties.
    target = torch.tensor([[1.], [0.]])
    assert multilabel_metrics(torch.tensor([[1.], [-1.]]), target)["map"] == 1
    assert multilabel_metrics(torch.tensor([[-1.], [1.]]), target)["map"] == 0.5
    assert multilabel_metrics(torch.zeros(2, 1), target)["map"] == 0.5
    result = {"status": "passed", "variants": checks, "shared_initial_weights_identical": True,
        "step_C_D_initial_weights_identical": True, "sample_order_identical": True, "multilabel_AP_hand_checks": True,
        "inference_cases_complete": expected_count,
        "limits": ["one training seed", "short training", "coverage-selected COCO subset", "frozen-coefficient probes are not a spatially varying stability proof"]}
    json_write(root/"audit.json", result)
    return result


def create_report(root):
    root = Path(root).resolve()
    manifest = read_json(root/"manifest.json")
    audit(root, manifest)
    config = manifest["config"]
    summaries = {v: read_json(root/v/"summary.json") for v in VARIANTS}
    series = {}
    for v in VARIANTS:
        with (root/v/"train.csv").open(encoding="utf-8") as stream:
            series[v] = list(csv.DictReader(stream))
    inference_directory = manifest.get("primary_inference_directory", "inference")
    inference = read_json(root/inference_directory/"summary.json")["aggregates"]
    plots = root/"plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "figure.facecolor": "white"})
    colors = plt.get_cmap("tab10").colors[:7]
    def save(fig, name):
        fig.savefig(plots/f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(plots/f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for index, v in enumerate(VARIANTS):
        color = colors[index]
        steps = [int(r["step"]) for r in series[v]]
        loss = np.array([float(r["loss"]) for r in series[v]])
        axes[0, 0].plot(steps, loss, color=color, linewidth=1.1, alpha=.85, label=LABELS[v])
        val = read_json(root/v/"validation.json")
        for ax, key, factor in ((axes[0, 1], "loss", 1), (axes[1, 0], "map", 100), (axes[1, 1], "micro_f1", 100)):
            ax.plot([r["step"] for r in val], [r[key]*factor for r in val], marker="o", markersize=3,
                    color=color, linestyle="--" if "multi" in v else "-", label=LABELS[v])
    for ax, title, ylabel in zip(axes.flat, ("Training BCE (raw, unsmoothed)", "Validation BCE", "Image-level multi-label mAP", "Micro F1 at threshold 0.5"),
                               ("BCE loss", "BCE loss", "mAP (%)", "F1 (%)")):
        ax.set(title=title, xlabel="Optimizer update", ylabel=ylabel)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"COCO diagnostic subset: {config['train_samples']} train / {config['val_samples']} validation | FP32, batch {config['batch_size']} | one seed", fontsize=12)
    save(fig, "loss_and_validation")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    stats = (("mean_train_step_ms", "Mean instrumented training step", "ms / batch"),
             ("peak_train_allocated_mib", "Peak training allocation", "MiB (PyTorch allocator)"),
             ("train_images_per_s", "Instrumented training throughput", "images / second"),
             ("energy", "Estimated whole-GPU training energy", "Wh (includes desktop GPU activity)"))
    for ax, (key, title, ylabel) in zip(axes.flat, stats):
        values = [summaries[v][key] if key != "energy" else summaries[v]["training_resources"]["estimated_device_energy_wh"] for v in VARIANTS]
        ax.bar(SHORT, [x if x is not None else 0 for x in values], color=colors, width=.7)
        ax.set(title=title, ylabel=ylabel)
        for i, value in enumerate(values):
            ax.annotate("N/A" if value is None else f"{value:.2f}", (i, value or 0), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax.margins(y=.2)
    fig.suptitle("L = local input; M = multi-direction input | optimizer, gradient checks and clipping included", fontsize=12)
    save(fig, "training_resources")
    ncols = len(config["inference"]["cases"])
    nrows = len(config["inference"]["precisions"])
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False, layout="constrained")
    for i, precision in enumerate(config["inference"]["precisions"]):
        for j, (size, batch) in enumerate(config["inference"]["cases"]):
            rows = [next(r for r in inference if (r["variant"], r["precision"], r["resolution"], r["batch_size"]) == (v, precision, size, batch)) for v in VARIANTS]
            values = np.array([r["p50_ms"] for r in rows])
            upper = np.array([r["p95_ms"] for r in rows])-values
            axes[i, j].bar(SHORT, values, yerr=np.stack((np.zeros_like(upper), upper)), capsize=3, color=colors)
            axes[i, j].set(title=f"{precision.upper()} | {size} x {size} | batch {batch}", ylabel="Batch latency (ms)")
            axes[i, j].tick_params(axis="x", labelsize=8)
    fig.suptitle("Median latency; whisker = empirical p95 (not confidence interval)\nFP32 is matched; BF16 autocast uses different internal scan precision", fontsize=12)
    save(fig, "inference_latency")
    fields = ["variant", "parameters", "first_train_loss", "final_train_loss", "initial_val_loss", "final_val_loss", "final_subset_map",
              "micro_f1", "mean_train_step_ms", "train_images_per_s", "peak_train_allocated_mib", "gpu_power_mean_w", "estimated_gpu_energy_wh"]
    comparison = []
    for v in VARIANTS:
        s = summaries[v]
        comparison.append({"variant": v, "parameters": s["model"]["parameters"], "first_train_loss": s["first_train_loss"],
            "final_train_loss": s["final_train_loss"], "initial_val_loss": s["initial_validation"]["loss"],
            "final_val_loss": s["final_validation"]["loss"], "final_subset_map": s["final_validation"]["map"],
            "micro_f1": s["final_validation"]["micro_f1"], "mean_train_step_ms": s["mean_train_step_ms"],
            "train_images_per_s": s["train_images_per_s"], "peak_train_allocated_mib": s["peak_train_allocated_mib"],
            "gpu_power_mean_w": s["training_resources"]["device_power_w_mean"],
            "estimated_gpu_energy_wh": s["training_resources"]["estimated_device_energy_wh"]})
    with (root/"comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)
    lines = ["# Mamba2D / FM-Mamba: historical seven-group pilot", "", "Status: all seven groups completed; log audit passed.", "",
        f"This run used {config['train_samples']} training images and {config['val_samples']} validation images from local COCO 2017 data. Each group trained for {config['steps']} steps with batch size {config['batch_size']}, {config['resolution']} x {config['resolution']} inputs, and FP32.",
        "This small-data pilot checks workflow and performance on subsets selected for class coverage. Reported mAP is mean image-level multilabel AP. Accuracy results describe this short, single-seed run.", "",
        "## Comparison setup", "",
        "- The original M2D-T backbone is the baseline. All groups use an 80-class output layer and BCEWithLogitsLoss, with matching settings for the remaining layers, stochastic depth, and head dropout.",
        "- All models are randomly initialized. SHA-256 checks verify identical shared outer weights, step projections, and initial C and D values. Data, sample order, training seed, and AdamW settings are shared: fixed learning rate 1e-4, weight decay 0.05, and gradient clipping at 1.0. EMA, CutMix, gradient accumulation, and learning-rate scheduling are disabled.",
        "- In this historical pilot, FM local input follows Equation 42 and multidirectional input follows Equation 21. Both use the first-order input coefficients from the corresponding draft revision.",
        "- For k=1...16, all three conditions use A1=A2=-k, with A0 equal to -k, -k^2, and -2.5k^2, respectively. FM uses fixed A coefficients; the original model learns A. The historical condition III protocol caps both step sizes at 0.99*ln(5)/16, approximately 0.099584.",
        "- Local-input B0(x) varies with position and input. In the historical multidirectional implementation, B1/B2 are position-independent trainable 16-dimensional vectors per channel. The injection is delta1*delta2*[B1*delta2*(x-x_above)+B2*delta1*(x-x_left)]. Inputs and states outside the boundary are zero; coefficients are evaluated at the current position. This historical formula retains the draft's additional delta factors.",
        "- The original model uses partial recomputation; the pilot FM CUDA implementation retains all forward states for backward propagation. Training memory and speed measurements characterize these execution policies.",
        "- Full images are resized to squares and normalized, with cropping and augmentation disabled. Labels are the union of all annotated categories, including crowd annotations. Images are decoded into a CPU cache before timing; training timings cover batch preparation and host-to-device transfer.", "",
        "## Training results", "", "| Model | Parameters | Final validation BCE | Subset mAP | ms/step | Peak memory MiB |", "|---|---:|---:|---:|---:|---:|"]
    for r in comparison:
        lines.append(f"| {LABELS[r['variant']]} | {r['parameters']:,} | {r['final_val_loss']:.5f} | {100*r['final_subset_map']:.2f}% | {r['mean_train_step_ms']:.2f} | {r['peak_train_allocated_mib']:.1f} |")
    lines += ["", "Training-step timings include gradient-norm collection, clipping, optimizer updates, and synchronization. Validation, log writes, and checkpoints have separate resource-log phases. Memory is the process-level peak PyTorch allocation. NVML memory, power, and utilization cover the whole GPU. Energy is estimated from sampled power, including desktop and other GPU activity.", "",
        "![Loss and validation metrics](plots/loss_and_validation.png)", "", "![Training resources](plots/training_resources.png)", "",
        "## Inference results", "", f"The table uses FP32 with inputs already on the GPU. p50/p95/p99, individual samples, repeated rounds, throughput, memory, and BF16 measurements are saved in {inference_directory}/summary.csv and samples.csv. Resource sampling runs separately from the primary timing measurements.", "",
        "| Model | Resolution | Batch | p50 ms | p95 ms | Sustained images/s | Peak MiB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in inference:
        if r["precision"] == "fp32":
            lines.append(f"| {LABELS[r['variant']]} | {r['resolution']} | {r['batch_size']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['sustained_images_per_s']:.1f} | {r['peak_allocated_mib']:.1f} |")
    lines += ["", "Inference measurements use repeated rounds with randomized model order. CUDA event spans include any idle time caused by CPU submission. Under BF16 autocast, the original scan uses BF16 while FM retains FP32; these measurements record each implementation's execution mode. End-to-end measurements also include JPEG decoding, preprocessing, and host-to-device transfer. Repeated reads generally use the system cache.", "",
        "![Inference latency](plots/inference_latency.png)", "", "## File index and audit checks", "",
        "- audit.json: training-step counts, per-step CSV/JSONL logs, validation records, resource samples, strict reload, sample order, and shared-initialization checks for all seven groups.",
        "- dataset_manifest.json / sample_schedule.json: image paths, image SHA-256 values, category mappings, preprocessed tensor hashes, and per-step sample indices.",
        "- Per-model train.csv / steps.jsonl: loss, learning rate, gradient norms, CPU batch preparation, host-to-device transfer, forward/backward/optimizer timings, throughput, and memory for every step.",
        "- Per-model resources.csv: CPU/RAM, GPU utilization, device memory, power, temperature, and clocks sampled approximately every 0.2 seconds; unsupported fields are empty.",
        "- Per-model validation.json: validation BCE, mAP, micro precision/recall/F1, macro F1, per-class AP, and sample counts.",
        "- Per-model step_diagnostics.json: step-size ranges, clipping fractions, and frozen-coefficient condition checks for six FM modules on a fixed diagnostic minibatch.",
        "- Per-model gradient_probes.json: gradient availability, finiteness, and post-clipping norms for each scan parameter.",
        "- Per-model checkpoint.pt / best_weights.pt: the final-step checkpoint with optimizer and RNG state, and weights with the lowest pilot validation BCE. Inference comparisons use the final-step checkpoint.",
        "- source_snapshot / packages.txt / git-diff.patch / manifest.json: source snapshots, dependency versions, Git state, machine environment, and execution order.",
        "- plots: PNG and SVG figures for the experiment record.", "",
        "Historical pilot command: `.\\run.ps1 run_comparison.py --config configs/fm-seven-pilot.json` in the original research workspace. Each run creates a new output directory. Dataset size, epoch budget, and seed repetitions are configured separately for full experiments.", ""]
    ablation_path = root/"monitor-ablation/result.json"
    if ablation_path.exists():
        ablation = read_json(ablation_path)["summary"]
        lines += ["## Timing-method review", "",
            "The first pass combined NVML/CPU resource sampling with inference timing and showed substantial latency tails. A follow-up used identical weights and inputs for the original model and FM condition I with local input, alternating monitoring on and off for three rounds with 180 timing samples per state.", "",
            "| Model | Resource sampling | p50 ms | p95 ms |", "|---|---|---:|---:|"]
        for row in ablation:
            lines.append(f"| {LABELS[row['variant']]} | {'Enabled' if row['monitor_enabled'] else 'Disabled'} | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} |")
        lines += ["", "Resource sampling affected tail latency on this machine. The primary inference results therefore use the separately repeated timing pass. Original monitored measurements remain in inference, and monitoring ablations remain in monitor-ablation. Training measurements cover the instrumented workflow, including monitoring, gradient checks, and clipping.", ""]
    (root/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return root/"REPORT.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    create_report(Path(parser.parse_args().directory))
