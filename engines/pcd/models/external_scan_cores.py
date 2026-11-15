"""Change only the scan inside the original Mamba2D parallel-local shell."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .kernels.external_scan import selective_scan

SCAN_VARIANTS = ('vision_mamba_scan', 'vmamba_scan', 'two_d_mamba_scan')
OLD_SCAN_KEYS = ('x_proj.', 'dt_projT.', 'dt_projL.', 'AT_log', 'AL_log', 'D')


class DirectionalScanCore(nn.Module):
    """Two raster directions or four row/column directions, with mean merge.

    There are no feature-channel input/output projections, convolutions,
    activation gates or normalization here. Only delta/B/C generation, the
    scan recurrence and its D skip live in this module.
    """
    def __init__(self, carrier, variant):
        super().__init__()
        if variant not in SCAN_VARIANTS:
            raise ValueError(variant)
        if carrier.d_state != 16 or carrier.expand_factor != 1 or carrier.double_scans:
            raise ValueError('The controlled experiment requires N=16, expansion=1, single M2D scan')
        self.variant = variant
        self.kind = 'two_d_mamba' if variant == 'two_d_mamba_scan' else 'vmamba'
        self.d_model = carrier.d_inner
        self.d_state, self.dt_rank = carrier.d_state, carrier.dt_rank
        self.directions = 2 if variant == 'vision_mamba_scan' else 4
        r, n = self.dt_rank, self.d_state
        old = carrier.x_proj.weight.detach()
        b = (old[2*r:2*r+n] + old[2*r+n:2*r+2*n]) / math.sqrt(2)
        c = old[-n:]
        projections, weights, biases, logs = [], [], [], []
        # All methods receive the same first two directional initializers.
        # Four-direction methods duplicate them for the reverse directions;
        # these are independent trainable parameters after initialization.
        for direction in range(self.directions):
            slot = direction % 2
            projection = carrier.dt_projT if slot == 0 else carrier.dt_projL
            projections.append(torch.cat((old[slot*r:(slot+1)*r], b, c)))
            weights.append(projection.weight.detach().clone())
            biases.append(projection.bias.detach().clone())
            logs.append((carrier.AT_log if slot == 0 else carrier.AL_log).detach().clone())
        self.x_proj_weight = nn.Parameter(torch.stack(projections))
        self.dt_proj_weight = nn.Parameter(torch.stack(weights))
        self.dt_proj_bias = nn.Parameter(torch.stack(biases))
        self.A_log = nn.Parameter(torch.stack(logs))
        self.D = nn.Parameter(carrier.D.detach().repeat(self.directions, 1))
        # Packing per-direction biases into a matrix must not turn the
        # original non-decayed biases into a weight-decayed projection.
        self.dt_proj_bias._no_weight_decay = True
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.scan = selective_scan

    def forward(self, x):
        batch, height, width, channels = x.shape
        length, k = height * width, self.directions
        spatial = x.permute(0, 3, 1, 2)
        row = spatial.flatten(2)
        if k == 2:
            xs = torch.stack((row, row.flip(-1)), dim=1)
        else:
            rowcol = torch.stack((row, spatial.transpose(2, 3).flatten(2)), dim=1)
            xs = torch.cat((rowcol, rowcol.flip(-1)), dim=1)
        projected = F.conv1d(xs.reshape(batch, k*channels, length),
                             self.x_proj_weight.reshape(-1, channels, 1), groups=k)
        dt, b, c = projected.reshape(batch, k, -1, length).split(
            (self.dt_rank, self.d_state, self.d_state), dim=2)
        dt = F.conv1d(dt.reshape(batch, k*self.dt_rank, length).contiguous(),
                      self.dt_proj_weight.reshape(k*channels, self.dt_rank, 1), groups=k)
        a = -self.A_log.float().exp().reshape(k*channels, self.d_state)
        if self.kind == 'two_d_mamba' and height != width:
            outputs=[]
            for direction in range(k):
                h,w = (height,width) if direction % 2 == 0 else (width,height)
                span = slice(direction*channels,(direction+1)*channels)
                outputs.append(self.scan(xs[:,direction],dt[:,span],a[span],
                                         b[:,direction:direction+1],c[:,direction:direction+1],
                                         self.D[direction],self.dt_proj_bias[direction],self.kind,h,w))
            ys=torch.stack(outputs,dim=1)
        else:
            ys=self.scan(xs.reshape(batch,k*channels,length),dt,a,b,c,self.D.flatten(),
                         self.dt_proj_bias.flatten(),self.kind,height,width)
            ys=ys.reshape(batch,k,channels,length)
        if k == 2:
            result=(ys[:,0]+ys[:,1].flip(-1))/2
        else:
            aligned=ys[:,:2]+ys[:,2:].flip(-1)
            result=(aligned[:,0]+aligned[:,1].reshape(batch,channels,width,height)
                    .transpose(2,3).flatten(2))/4
        return result.transpose(1,2).reshape(batch,height,width,channels)


class ScanOnlyM2DBlock(nn.Module):
    """Retain the exact original projections, GELUs and parallel local branch."""
    def __init__(self, carrier, variant):
        super().__init__()
        if carrier.local_path is None:
            raise ValueError('Expected the original parallel local branch')
        self.scan_start = carrier.scan_start
        self.d_model, self.d_inner = carrier.d_model, carrier.d_inner
        self.d_state, self.expand_factor = carrier.d_state, carrier.expand_factor
        self.local_norm, self.local_path = carrier.local_norm, carrier.local_path
        self.in_proj, self.out_proj = carrier.in_proj, carrier.out_proj
        self.act1, self.act2 = carrier.act1, carrier.act2
        self.scan_core=DirectionalScanCore(carrier,variant)

    def ssm(self,x):
        return self.scan_core(x)

    def forward(self,x):
        from .mamba2d import M2DBlock
        return M2DBlock.forward(self,x)


def replace_scan_cores(model,variant):
    from .mamba2d import M2DBlock
    if list(model.backbone.token_mixer) != ['2D_local','2D_local','RopeAttention','RopeAttention']:
        raise ValueError('Only the original 2M2A carrier is supported')
    cores=[(name,module) for name,module in model.named_modules() if isinstance(module,M2DBlock)]
    if len(cores)!=6 or sorted(m.d_inner for _,m in cores)!=[64]*3+[128]*3:
        raise ValueError('Expected six original 64/128-channel Mamba2D blocks')
    for name,old in cores:
        model.set_submodule(name,ScanOnlyM2DBlock(old,variant))
    model.backbone.token_mixer=[variant,variant,'RopeAttention','RopeAttention']
    return [name for name,_ in cores]
