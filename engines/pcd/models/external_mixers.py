"""Official visual-Mamba mixing operators in the fixed 2M2A carrier.

Vim uses the pinned official BiMamba-v2 initializer. Its fused inner function
is expressed as the same convolution, projections, CUDA scan and SiLU gate.
SS2D follows VMamba-T's v05_noz operator; 2DMamba changes its scan kernel only.
These are hybrid-backbone experiments, not reproductions of the original
full Vim/VMamba/2DVMamba image classifiers. See configs/groups.json for group definitions.
"""
from functools import lru_cache
import importlib.util
import math

import torch
from torch import nn
from torch.nn import functional as F

from .kernels.external_scan import UPSTREAM, selective_scan


EXTERNAL_VARIANTS = ("vision_mamba", "vmamba", "two_d_mamba")


@lru_cache(None)
def vim_class():
    path = UPSTREAM / "vim/mamba-1p1p1/mamba_ssm/modules/mamba_simple.py"
    spec = importlib.util.spec_from_file_location("pinned_vim_mamba_simple", path)
    module = importlib.util.module_from_spec(spec)
    source = path.read_text(encoding="utf-8").replace(
        "causal_conv1d_fn, causal_conv1d_update = None\n",
        "causal_conv1d_fn, causal_conv1d_update = None, None\n")
    source = source.replace(
        "selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None",
        "selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.Mamba


class VisionMambaMixer(nn.Module):
    """BiMamba-v2: independent forward/backward scans, shared gate, mean merge."""
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.core = vim_class()(d_model=d_model, d_state=d_state, expand=expand,
                                bimamba_type="v2", if_divide_out=True)
        self.scan = selective_scan

    def branch(self, xz, reverse=False):
        core = self.core
        suffix = "_b" if reverse else ""
        if reverse:
            xz = xz.flip(-1)
        x, z = xz.chunk(2, dim=1)
        length = x.shape[-1]
        x = F.silu(getattr(core, "conv1d" + suffix)(x)[..., :length])
        projected = getattr(core, "x_proj" + suffix)(x.transpose(1, 2))
        dt, B, C = projected.split([core.dt_rank, core.d_state, core.d_state], dim=-1)
        dt_proj = getattr(core, "dt_proj" + suffix)
        dt = F.linear(dt, dt_proj.weight).transpose(1, 2).contiguous()
        A_log = core.A_b_log if reverse else core.A_log
        y = self.scan(x, dt, -A_log.float().exp(), B.transpose(1, 2).unsqueeze(1),
                      C.transpose(1, 2).unsqueeze(1), getattr(core, "D" + suffix), dt_proj.bias)
        y = y * F.silu(z)
        return y.flip(-1) if reverse else y

    def forward(self, x):
        batch, height, width, channels = x.shape
        xz = self.core.in_proj(x.reshape(batch, height * width, channels)).transpose(1, 2)
        y = (self.branch(xz) + self.branch(xz, reverse=True)) * 0.5
        return self.core.out_proj(y.transpose(1, 2)).reshape(batch, height, width, channels)


def _dt_init(rank, inner):
    layer = nn.Linear(rank, inner, bias=True)
    scale = rank ** -0.5
    nn.init.uniform_(layer.weight, -scale, scale)
    dt = torch.exp(torch.rand(inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp_min(1e-4)
    with torch.no_grad():
        layer.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
    return layer


class SS2DMixer(nn.Module):
    """VMamba-T v05_noz, optionally using the official 2DMamba scan.

Default d_state=1 and expansion=1 follow both official tiny configurations.
Four directions have independent scan projections and are summed, not averaged.
"""
    def __init__(self, d_model, kind="vmamba", d_state=1, expand=1):
        super().__init__()
        self.kind = kind
        self.d_model, self.d_state = d_model, d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, 3, padding=1,
                                groups=self.d_inner, bias=False)
        x_proj = [nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False) for _ in range(4)]
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in x_proj]))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        dt_projs = [_dt_init(self.dt_rank, self.d_inner) for _ in range(4)]
        self.dt_projs_weight = nn.Parameter(torch.stack([layer.weight for layer in dt_projs]))
        self.dt_projs_bias = nn.Parameter(torch.stack([layer.bias for layer in dt_projs]))
        self.A_logs = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32).log()
                                   .repeat(4 * self.d_inner, 1))
        self.Ds = nn.Parameter(torch.ones(4 * self.d_inner))
        self.A_logs._no_weight_decay = True
        self.Ds._no_weight_decay = True
        self.scan = selective_scan

    def forward(self, x):
        x = F.silu(self.conv2d(self.in_proj(x).permute(0, 3, 1, 2).contiguous()))
        batch, channels, height, width = x.shape
        length = height * width
        rowcol = torch.stack((x.flatten(2), x.transpose(2, 3).flatten(2)), dim=1)
        xs = torch.cat((rowcol, rowcol.flip(-1)), dim=1)
        projected = F.conv1d(xs.reshape(batch, 4 * channels, length),
                             self.x_proj_weight.reshape(-1, channels, 1), groups=4)
        dt, B, C = projected.reshape(batch, 4, -1, length).split(
            [self.dt_rank, self.d_state, self.d_state], dim=2)
        dt = F.conv1d(dt.reshape(batch, 4 * self.dt_rank, length).contiguous(),
                      self.dt_projs_weight.reshape(4 * channels, self.dt_rank, 1), groups=4)
        A = -self.A_logs.float().exp()
        if self.kind == "two_d_mamba" and height != width:
            # Column-major directions have a transposed grid, as upstream's
            # SPERATE_HW branch requires for rectangular feature maps.
            parts = []
            for direction in range(4):
                h, w = (height, width) if direction % 2 == 0 else (width, height)
                span = slice(direction * channels, (direction + 1) * channels)
                parts.append(self.scan(xs[:, direction], dt[:, span], A[span],
                                       B[:, direction:direction+1], C[:, direction:direction+1],
                                       self.Ds[span], self.dt_projs_bias[direction], self.kind, h, w))
            ys = torch.stack(parts, dim=1)
        else:
            ys = self.scan(xs.reshape(batch, 4 * channels, length), dt, A, B, C,
                           self.Ds, self.dt_projs_bias.flatten(), self.kind, height, width)
            ys = ys.reshape(batch, 4, channels, length)
        aligned = ys[:, :2] + ys[:, 2:].flip(-1)
        y = aligned[:, 0].reshape(batch, channels, height, width)
        y = y + aligned[:, 1].reshape(batch, channels, width, height).transpose(2, 3)
        y = self.out_norm(y.permute(0, 2, 3, 1).contiguous()).to(x.dtype)
        return self.out_proj(y)


def replace_first_two_stages(model, variant):
    from .mamba2d import M2DBlock
    if variant not in EXTERNAL_VARIANTS:
        raise ValueError(variant)
    if list(model.backbone.token_mixer) != ["2D_local", "2D_local", "RopeAttention", "RopeAttention"]:
        raise ValueError("External baseline adapters require the original 2M2A carrier")
    paths = [(name, module) for name, module in model.named_modules() if isinstance(module, M2DBlock)]
    if len(paths) != 6 or sorted(module.d_model for _, module in paths) != [64]*3 + [128]*3:
        raise ValueError("Expected the six 64/128-channel mixers in stages 1 and 2")
    for name, old in paths:
        mixer = VisionMambaMixer(old.d_model) if variant == "vision_mamba" else SS2DMixer(old.d_model, variant)
        model.set_submodule(name, mixer)
    model.backbone.token_mixer = [variant, variant, "RopeAttention", "RopeAttention"]
    return [name for name, _ in paths]
