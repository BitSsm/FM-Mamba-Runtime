"""Manifest and status helpers for the 3-layout x 9-variant experiment."""
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import time

BASE_VARIANTS = ['mamba2d'] + [f'fm_c{c}_{f}' for c in (1, 2, 3) for f in ('local', 'multi')]
ARCHITECTURE_VERSION = 'hybrid_vmamba_serial_20261016'
CAP_VARIANTS = ['fm_c3_local', 'fm_c3_multi']
LAYOUTS = {
    '2M2A': 'configs/M2D-T-ImageNet.yaml',
    '3M1A': 'configs/M2D-T-3M1A-backbone.yaml',
    '4M0A': 'configs/M2D-T-4M0A-backbone.yaml',
}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def cursor(group):
    """Checkpoint is authoritative; progress JSON is only for presentation."""
    import torch
    folder = Path(group['run']) / group['variant']
    path = folder / 'checkpoint.pt'
    if not path.exists():
        path = folder / 'previous_checkpoint.pt'
    if not path.exists():
        return (0, 0, 0)
    value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    return value['epoch'], value['next_batch'], value['step']


def status(root):
    root = Path(root)
    manifest = read(root / 'manifest.json')
    queue = read(root / 'queue_status.json', {'status': 'prepared'})
    rows = []
    for group in manifest['groups']:
        folder = Path(group['run']) / group['variant']
        progress = read(folder / 'progress.json', {})
        epochs = [read(p) for p in sorted((folder / 'epochs').glob('*/metrics.json'))]
        best = max(epochs, key=lambda x: x['map']) if epochs else {}
        spec = read(folder / 'model_spec.json', {})
        rows.append(dict(group, epochs_completed=progress.get('epochs_completed', 0),
                         step=progress.get('step', 0), phase=progress.get('status', 'waiting'),
                         best_epoch=best.get('epoch'), best_map=best.get('map'),
                         best_micro_f1=best.get('micro_f1'), best_macro_f1=best.get('macro_f1'),
                         parameters=spec.get('parameters'),
                         mean_step_ms=(sum(e['train_wall_s'] for e in epochs) * 1000 /
                                       epochs[-1]['step']) if epochs else None))
    value = {'updated': now(), 'queue': queue, 'pilot': manifest['pilot'],
             'target_epochs': manifest['target_epochs'], 'groups': rows}
    write(root / 'status.json', value)
    esc = html.escape
    lines = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
             '<title>FM-Mamba：三种结构 × 九组实验</title>',
             '<style>body{font:16px/1.6 system-ui;margin:32px;color:#172033;background:#f5f7fb}'
             'table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px 12px;border-bottom:1px solid #ddd;text-align:right}'
             'th:first-child,td:first-child{text-align:left}th{background:#e4ebf4}small{color:#556}a{color:#2563a7}</style>',
             '<h1>三种结构 × 九组实验</h1>',
             f'<p>{"短程流程验证" if manifest["pilot"] else "正式训练"} · 每组 {manifest["target_epochs"]} 轮 · '
             f'状态：{esc(queue["status"])} · 更新：{esc(value["updated"])}</p>',
             '<p>2M2A / 3M1A / 4M0A 分别含 6 / 15 / 18 个 Mamba 块。工况 3 默认上限为 0.99 ln(5)/16，'
             'cap090 两组为 0.90 ln(5)/16。共享初始化在同一结构内逐项核对；跨结构采用同一随机种子。</p>',
             '<p>结构修订：前两阶段保留各组原有扫描及并联局部分支；第三阶段或第三、四阶段采用 '
             'Linear → DWConv3 → SiLU → 对应扫描核心 → LayerNorm → Linear，线性层和卷积均无偏置。</p>',
             '<p>COCO 80 类图片多标签分类；35 轮实验保留原 40 轮学习率计划。历史 7 组复用；新增 20 组从头训练。'
             '验证指标不代表检测 AP。单种子结果不能估计种子间波动。推理会在训练结束后统一重测 27 组最佳权重。</p>',
             '<table><thead><tr><th>实验</th><th>来源</th><th>轮数</th><th>步数</th><th>最佳轮</th><th>mAP %</th>'
             '<th>micro-F1 %</th><th>macro-F1 %</th><th>参数 M</th></tr></thead><tbody>']
    fmt = lambda v, scale=1: '—' if v is None else f'{v * scale:.3f}'
    for row in rows:
        lines.append('<tr><td>' + esc(row['id']) + '</td><td>' + ('复用' if row['reuse'] else '新增') + '</td>'
                     f'<td>{row["epochs_completed"]}</td><td>{row["step"]}</td><td>{row["best_epoch"] or "—"}</td>'
                     f'<td>{fmt(row["best_map"], 100)}</td><td>{fmt(row["best_micro_f1"], 100)}</td>'
                     f'<td>{fmt(row["best_macro_f1"], 100)}</td><td>{fmt(row["parameters"], 1e-6)}</td></tr>')
    lines += ['</tbody></table>', '<p><small>各组轮数未齐时，表中阶段性指标仅用于观察进度；训练耗时、显存、功耗和逐批日志保存在各组目录。</small></p>']
    (root / 'STATUS.html').write_text('\n'.join(lines), encoding='utf-8')
    return value
