"""Public model loader for the 14 experiment groups.

The historical and PCD engines intentionally keep distinct module namespaces.
Use separate processes when switching engines (the CLI does this naturally).
"""
import json
import os
from pathlib import Path
import sys
from fm_mamba_runtime import engine_directory as binary_directory

ROOT = Path(__file__).resolve().parent
_ACTIVE = None

def engine_directory(name):
    binary_directory(name)  # Validate installed runtime and engine name.
    return ROOT / 'engines' / name


def groups():
    return json.loads((ROOT / 'configs/groups.json').read_text(encoding='utf8'))['groups']


def group(identifier):
    return next(g for g in groups() if g['id'] == identifier)


def configuration(identifier):
    info = group(identifier)
    config = json.loads((ROOT / info['training_config']).read_text(encoding='utf8'))
    config['model_config'] = str(engine_directory(info['engine']) / config['model_config'])
    return config


def activate(identifier):
    global _ACTIVE
    info = group(identifier)
    if _ACTIVE is not None and _ACTIVE != info['engine']:
        raise RuntimeError('Switch engines in a fresh Python process to avoid legacy module-name collisions.')
    engine = engine_directory(info['engine'])
    if _ACTIVE is None:
        sys.path.insert(0, str(engine))
        sys.path.append(str(binary_directory(info['engine'])))
        _ACTIVE = info['engine']
    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / '.torch_extensions' / info['engine']))
    return info, engine


def build_model(identifier, checkpoint=None, *, device='cuda', config=None):
    import torch
    info, engine = activate(identifier)
    from comparison_support import build_variant
    from full_worker import configure_torch
    config = configuration(identifier) if config is None else config
    configure_torch(config)
    model, spec = build_variant(info['variant'], config)
    if spec['parameters'] != info['parameters']:
        raise RuntimeError('Parameter count does not match the selected group.')
    if checkpoint:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved.get('variant', info['variant']) != info['variant']:
            raise ValueError('Checkpoint variant does not match the selected group.')
        model.load_state_dict(saved.get('state_dict', saved), strict=True)
        del saved
    model.to(device)
    if info['engine'] == 'pcd':
        from optimized_serial_runtime import apply_runtime
        spec['optimized_runtime'] = apply_runtime(model, config, info['variant'])
    spec['public_group'] = identifier
    return model, spec


def inference_graph(model, example):
    if _ACTIVE != 'pcd':
        raise ValueError('CUDA Graph is enabled only for the three PCD groups.')
    from runtime_graphs import InferenceGraph
    return InferenceGraph(model, example)
