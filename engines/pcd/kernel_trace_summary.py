from collections import Counter,defaultdict
import re
from stage_matrix_common import read

def category(name):
    if any(k in name for k in ('block_plane','tile_wave','tile_group')):return 'FM_forward_plane_or_tile'
    if 'ring_wave' in name:return 'FM_forward_ring'
    if 'fused_grad_wave' in name:return 'FM_backward_with_B_or_BC'
    if 'rolling_wave' in name:return 'FM_backward_ring'
    if any(k in name for k in ('wave_bc','bc_reduce','partial_reduce','c_reduce')):return 'FM_BC_gradient_reduction'
    if 'fused_wave' in name:
        # Windows CUPTI may preserve Itanium CUDA symbols instead of demangling.
        # The first boolean template argument is backward, and the last is readout.
        if name.startswith('_Z'):
            flags=re.findall(r'Lb([01])E',name.split('Ev',1)[0])
            if flags:
                if flags[0]=='1':return 'FM_backward_full'
                return 'FM_forward_full' if flags[-1]=='1' else 'FM_recompute'
        m=re.search(r'fused_wave<([^>]+)>',name)
        if m:
            flags=[x.strip() for x in m[1].split(',')]
            if flags[1] in ('1','true'):return 'FM_backward_full'
            return 'FM_forward_full' if flags[-1] in ('1','true') else 'FM_recompute'
        return 'FM_scan_unclassified'
    if 'wf_fwd' in name:return 'Mamba2D_forward_or_recompute'
    if 'wf_bwd' in name:return 'Mamba2D_backward'
    lower=name.lower()
    if 'selective_scan' in lower:
        return 'external_scan_backward' if 'bwd' in lower else 'external_scan_forward'
    if any(k in lower for k in ('attention','flash','fmha')):return 'attention'
    if any(k in lower for k in ('gemm','gemv','cutlass','cublas','mma_')):return 'matrix_or_convolution'
    if any(k in lower for k in ('cudnn','conv','winograd','implicit_gemm')):return 'matrix_or_convolution'
    if any(k in lower for k in ('adam','multi_tensor_apply')):return 'optimizer_or_multi_tensor'
    return 'other_network_kernels'

def summarize(path,repeats):
    data=read(path);events=data['traceEvents']
    kernels=[e for e in events if e.get('cat')=='kernel' and 'dur' in e]
    assert kernels,'CUDA events missing: cannot replace them with host launch counts'
    categories=defaultdict(lambda:dict(count=0,total_ms=0.))
    names={}
    for e in kernels:
        tag=category(e['name']);c=categories[tag];c['count']+=1;c['total_ms']+=e['dur']/1000
        entry=names.setdefault(e['name'],dict(name=e['name'],category=tag,count=0,total_ms=0.,min_us=e['dur'],max_us=e['dur'],launch_metadata=[]))
        entry['count']+=1;entry['total_ms']+=e['dur']/1000
        entry['min_us']=min(entry['min_us'],e['dur']);entry['max_us']=max(entry['max_us'],e['dur'])
        meta={k:v for k,v in e.get('args',{}).items() if any(s in k.lower() for s in ('grid','block','register','shared','stream','device'))}
        if meta not in entry['launch_metadata']:entry['launch_metadata'].append(meta)
    total=sum(e['dur'] for e in kernels)/1000
    for c in categories.values():
        c.update(calls_per_iteration=c['count']/repeats,ms_per_iteration=c['total_ms']/repeats,share_of_kernel_sum_pct=100*c['total_ms']/total)
    for e in names.values():e.update(calls_per_iteration=e['count']/repeats,ms_per_iteration=e['total_ms']/repeats,share_of_kernel_sum_pct=100*e['total_ms']/total,mean_us=e['total_ms']*1000/e['count'])
    host=Counter(e['name'] for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'dur' in e)
    runtime_graph=sum(n for k,n in host.items() if k.startswith('cudaGraphLaunch'))
    driver_graph=sum(n for k,n in host.items() if k.startswith('cuGraphLaunch'))
    devices=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and 'dur' in e]
    intervals=sorted((e['ts'],e['ts']+e['dur']) for e in devices);start,end=intervals[0];busy=0.
    for left,right in intervals[1:]:
        if left>end:busy+=end-start;start,end=left,right
        else:end=max(end,right)
    busy+=end-start;span=max(b for a,b in intervals)-intervals[0][0]
    return dict(repeats=repeats,kernel_calls=len(kernels),kernel_calls_per_iteration=len(kernels)/repeats,
        kernel_duration_sum_ms=total,kernel_ms_per_iteration=total/repeats,categories=dict(categories),
        kernel_details=sorted(names.values(),key=lambda r:-r['total_ms']),host_cuda_calls=dict(host),
        host_graph_launches_per_iteration=(runtime_graph if runtime_graph else driver_graph)/repeats,
        graph_runtime_api_calls=runtime_graph,graph_driver_api_calls=driver_graph,
        graph_count_basis='Runtime API when present; otherwise driver API. Nested API events are not added twice.',
        device_busy_union_ms=busy/1000,device_span_ms=span/1000,gap_fraction=1-busy/span,
        memcpy_events=sum(e.get('cat')=='gpu_memcpy' for e in devices),memset_events=sum(e.get('cat')=='gpu_memset' for e in devices),
        scope='Instrumented timeline only; kernel-duration sum is not whole-step latency. Device gaps are not attributed solely to CPU submission.',
        classification='Name-based categories. Mamba2D forward/recompute are combined when kernel names cannot distinguish them; other_network_kernels is explicit remainder.',
        hardware_counter_note='Launch metadata is from the trace when present, not new DRAM/L2 hardware-counter measurements.')
