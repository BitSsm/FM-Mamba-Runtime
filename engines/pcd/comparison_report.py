"""Audit pilot outputs and render scientific plots plus a Chinese run report."""
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
    lines = ["# Mamba2D / FM-Mamba 七组短程试验", "", "状态：七组完成，日志审计通过。", "",
        f"本轮使用本机 COCO 2017 的 {config['train_samples']} 张训练图、{config['val_samples']} 张验证图，各组均训练 {config['steps']} 步，batch={config['batch_size']}，输入 {config['resolution']}×{config['resolution']}，FP32。",
        "这是小样本流程与性能试验。子集按类别覆盖抽取，并非完整 COCO 验证集；以下 mAP 是图像级多标签分类 AP 的均值，不是目标检测框 AP，也不是 ImageNet 准确率。单次短训练不能说明最终精度或显著优劣。", "",
        "## 对比设置", "",
        "- 原版 M2D-T 骨干作基线；所有组把最后一层改成 80 类，采用同一 BCEWithLogitsLoss。网络其他层、随机深度和 head dropout 保持配置一致。",
        "- 全部随机初始化；共有外层权重、步长投影、C 与 D 的初始值经 SHA-256 检查完全一致。相同数据、顺序、训练 seed、AdamW、固定学习率 1e-4、weight decay 0.05、梯度裁剪 1.0；无 EMA、CutMix、梯度累积和学习率调度。",
        "- FM 单一就地输入对应公式 42；多方向输入对应公式 21。均按草稿标红的一阶输入系数实现，未把积分形式当成另一种输入结构。",
        "- 令 k=1…16：三工况 A1=A2=-k，A0 分别为 -k、-k²、-2.5k²。FM 的 A 固定；原版的 A 可学习。工况 3 对两方向步长硬截断至 0.99·ln(5)/16，约 0.099584。",
        "- 单一输入 B0(x) 随位置/输入变化；多方向 B1/B2 是每通道 16 维可学习向量，位置无关。多方向注入为 Δ1Δ2[B1Δ2(x−x上)+B2Δ1(x−x左)]，边界外输入和状态为零，系数均取当前位置。这里忠实保留草稿额外的 Δ 因子，没有自行替换成除以 Δ 的差分公式。",
        "- 原版沿用 partial 重计算；当前 FM CUDA 保存全部状态供反向使用。因此训练显存与速度反映当前实现，不能直接解释为算法固有复杂度优势。",
        "- 图片保留全图，缩放至正方形并归一化，不裁剪、不做数据增强；类别标签为该图所有标注类别的并集，包含 crowd。图片预先解码到 CPU 缓存，训练计时包含 batch 准备和 H2D，不包含首次磁盘解码。", "",
        "## 训练结果", "", "| 模型 | 参数数 | 最终验证 BCE | 子集 mAP | 每步 ms | 峰值显存 MiB |", "|---|---:|---:|---:|---:|---:|"]
    for r in comparison:
        lines.append(f"| {LABELS[r['variant']]} | {r['parameters']:,} | {r['final_val_loss']:.5f} | {100*r['final_subset_map']:.2f}% | {r['mean_train_step_ms']:.2f} | {r['peak_train_allocated_mib']:.1f} |")
    lines += ["", "训练步长计时包含梯度范数采集、裁剪、优化器和同步；验证、日志写盘及 checkpoint 单列在资源日志中。显存是本进程 PyTorch 分配峰值；NVML 的显存/功率/利用率是整张 GPU 的数据。功耗积分为采样估计，包含桌面或其他进程的 GPU 活动。", "",
        "![loss 与验证指标](plots/loss_and_validation.png)", "", "![训练资源](plots/training_resources.png)", "",
        "## 推理结果", "", f"下表为 FP32；输入已在 GPU。p50/p95/p99、逐次样本、重复轮次、吞吐、显存以及 BF16 数据均见 {inference_directory}/summary.csv 和 samples.csv。精确计时期间关闭本脚本的资源采样，资源数据在单独的测量中采集。", "",
        "| 模型 | 分辨率 | batch | p50 ms | p95 ms | 持续吞吐 张/秒 | 峰值 MiB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in inference:
        if r["precision"] == "fp32":
            lines.append(f"| {LABELS[r['variant']]} | {r['resolution']} | {r['batch_size']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['sustained_images_per_s']:.1f} | {r['peak_allocated_mib']:.1f} |")
    lines += ["", "推理按随机模型顺序重复测量。CUDA event 是整个时间跨度，可能包含 CPU 提交造成的空闲，不等同于 kernel 执行时间之和。BF16 下原版扫描使用 BF16、FM 扫描保留 FP32，所以 BF16 仅作为各实现运行模式的补充记录。端到端指标另含 JPEG 解码、预处理及 H2D，重复读图通常命中系统缓存。", "",
        "![推理延迟](plots/inference_latency.png)", "", "## 文件索引与通过条件", "",
        "- audit.json：七组训练步数、逐步 CSV/JSONL、验证记录、资源采样、严格重载、样本顺序与共享初始化检查。",
        "- dataset_manifest.json / sample_schedule.json：样本路径、图片 SHA-256、类别映射、预处理后张量哈希与每一步样本索引。",
        "- 各模型目录 train.csv / steps.jsonl：每步 loss、学习率、梯度范数、CPU batch 准备、H2D、前向、反向、优化器时间、吞吐和显存。",
        "- 各模型目录 resources.csv：约每 0.2 秒记录 CPU/RAM、GPU 利用率、设备显存、功率、温度、时钟；不支持的字段记为空值。",
        "- 各模型目录 validation.json：验证 BCE、mAP、微平均 P/R/F1、宏平均 F1、逐类 AP 和样本数。",
        "- 各模型目录 step_diagnostics.json：固定诊断小批次上 6 个 FM 模块的步长范围、截断比例和冻结系数条件检查；不代表全数据/空间变化系统的稳定性证明。",
        "- 各模型目录 gradient_probes.json：逐个扫描参数的梯度是否存在、是否有限，以及裁剪后的范数。",
        "- 各模型目录 checkpoint.pt / best_weights.pt：最后一步含优化器和 RNG 状态的 checkpoint、短程中验证 BCE 最低的权重；推理比较一律使用最后一步。",
        "- source_snapshot / packages.txt / git-diff.patch / manifest.json：源代码快照、依赖版本、Git 状态、机器环境与运行顺序。",
        "- plots 中同时保存 PNG 和 SVG，可进一步用于整理实验记录。", "",
        "复跑：在 repo 中运行 `.\\run.ps1 run_comparison.py --config configs/fm-seven-pilot.json`。每次创建新输出目录；正式训练仍需另定数据规模、轮数与多随机种子方案。", ""]
    ablation_path = root/"monitor-ablation/result.json"
    if ablation_path.exists():
        ablation = read_json(ablation_path)["summary"]
        lines += ["## 计时方法复核", "",
            "首轮同时进行 NVML/CPU 资源采样和推理计时，观察到明显长尾。随后用同样权重、输入，对原版和 FM 工况 1 单一输入做 3 轮交替开关采样的对照，每种状态 180 次计时，结果如下。", "",
            "| 模型 | 资源采样 | p50 ms | p95 ms |", "|---|---|---:|---:|"]
        for row in ablation:
            lines.append(f"| {LABELS[row['variant']]} | {'开启' if row['monitor_enabled'] else '关闭'} | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} |")
        lines += ["", "这说明本机资源采样会明显影响尾部延迟。因此上述主推理结果采用重新测得的独立计时数据；原始带采样数据保留在 inference 目录，开关对照保留在 monitor-ablation 目录。训练资源数据仍属于带监测、梯度检查和裁剪的实际训练流程计时，不当作无监测的纯计算时间。", ""]
    (root/"REPORT.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    return root/"REPORT.zh-CN.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    create_report(Path(parser.parse_args().directory))
