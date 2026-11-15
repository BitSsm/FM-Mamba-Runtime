"""Precompiled VMamba / 2DMamba scan adapters."""
from functools import lru_cache
import os
from pathlib import Path

import torch

UPSTREAM = Path(__file__).resolve().parents[4] / 'THIRD_PARTY/vision_baselines'


@lru_cache(None)
def load_scan_kernel(kind):
    from fm_mamba_runtime import load_native
    return load_native('external_' + kind + '_scan')


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, delta_bias, kind, height, width):
        kernel = load_scan_kernel(kind)
        u, delta, A, B, C, D, delta_bias = [t.contiguous() for t in (u, delta, A, B, C, D, delta_bias)]
        if kind == "vmamba":
            out, state, *_ = kernel.fwd(u, delta, A, B, C, D, delta_bias, True, 1, True)
        else:
            out, state, *_ = kernel.fwd(u, delta, A, B, C, D, None, delta_bias, True, height, width, True)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, state)
        ctx.kind, ctx.height, ctx.width = kind, height, width
        return out

    @staticmethod
    def backward(ctx, grad):
        u, delta, A, B, C, D, delta_bias, state = ctx.saved_tensors
        kernel = load_scan_kernel(ctx.kind)
        if ctx.kind == "vmamba":
            grads = kernel.bwd(u, delta, A, B, C, D, delta_bias, grad.contiguous(), state, True, 1)
        else:
            grads = kernel.bwd(u, delta, A, B, C, D, None, delta_bias, grad.contiguous(), state,
                               None, None, True, False, ctx.height, ctx.width)
        return (*grads[:7], None, None, None)


def selective_scan(u, delta, A, B, C, D, delta_bias, kind="vmamba", height=1, width=None):
    if not u.is_cuda:
        raise ValueError("Official external scan kernels require CUDA")
    if width is None:
        width = u.shape[-1]
    if height * width != u.shape[-1]:
        raise ValueError("Scan shape does not match sequence length")
    # 2DMamba's official kernel has float/half, but no bfloat16 specialization.
    # Preserve FP32 recurrence in BF16 mixed-precision tests and record this.
    if kind == "two_d_mamba" and u.dtype == torch.bfloat16:
        u, delta, B, C = [t.float() for t in (u, delta, B, C)]
    return _Scan.apply(u, delta, A.float(), B, C, D.float(), delta_bias.float(), kind, height, width)
