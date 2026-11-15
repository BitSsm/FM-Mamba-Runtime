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
             '<title>FM-Mamba: three architectures x nine variants</title>',
             '<style>body{font:16px/1.6 system-ui;margin:32px;color:#172033;background:#f5f7fb}'
             'table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px 12px;border-bottom:1px solid #ddd;text-align:right}'
             'th:first-child,td:first-child{text-align:left}th{background:#e4ebf4}small{color:#556}a{color:#2563a7}</style>',
             '<h1>Three architectures x nine variants</h1>',
             f'<p>{"Short workflow validation" if manifest["pilot"] else "Full training"} · {manifest["target_epochs"]} epochs per group · '
             f'Status: {esc(queue["status"])} · Updated: {esc(value["updated"])}</p>',
             '<p>2M2A / 3M1A / 4M0A contain 6 / 15 / 18 Mamba blocks, respectively. In this historical matrix, condition III uses a default cap of 0.99 ln(5)/16; '
             'the two cap090 variants use 0.90 ln(5)/16. Shared initialization is checked within each architecture; all architectures use the same random seed.</p>',
             '<p>Architecture revision: the first two stages retain each group\'s scan and parallel local branch. Stage 3, or stages 3 and 4, use '
             'Linear → DWConv3 → SiLU → scan core → LayerNorm → Linear, with bias-free linear and convolution layers.</p>',
             '<p>COCO 80-class image-level multilabel classification. The 35-epoch runs retain the original 40-epoch learning-rate schedule. Seven historical runs are reused; 20 additional runs start from scratch. '
             'The table reports classification metrics from one seed. The best checkpoints from all 27 groups are benchmarked after training.</p>',
             '<table><thead><tr><th>Experiment</th><th>Source</th><th>Epochs</th><th>Steps</th><th>Best epoch</th><th>mAP %</th>'
             '<th>micro-F1 %</th><th>macro-F1 %</th><th>Parameters M</th></tr></thead><tbody>']
    fmt = lambda v, scale=1: '—' if v is None else f'{v * scale:.3f}'
    for row in rows:
        lines.append('<tr><td>' + esc(row['id']) + '</td><td>' + ('Reused' if row['reuse'] else 'New') + '</td>'
                     f'<td>{row["epochs_completed"]}</td><td>{row["step"]}</td><td>{row["best_epoch"] or "—"}</td>'
                     f'<td>{fmt(row["best_map"], 100)}</td><td>{fmt(row["best_micro_f1"], 100)}</td>'
                     f'<td>{fmt(row["best_macro_f1"], 100)}</td><td>{fmt(row["parameters"], 1e-6)}</td></tr>')
    lines += ['</tbody></table>', '<p><small>Interim metrics track progress while epoch counts differ across groups. Each group directory stores training times, memory, power, and per-batch logs.</small></p>']
    (root / 'STATUS.html').write_text('\n'.join(lines), encoding='utf-8')
    return value
