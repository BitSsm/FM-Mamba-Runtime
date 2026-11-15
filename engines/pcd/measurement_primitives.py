import gc
import time
from runtime_graphs import InferenceGraph
def select(model,ring=False): pass

def stats(samples):
    import numpy as np
    return dict(median=float(np.median(samples)), p95=float(np.percentile(samples,95)), p99=float(np.percentile(samples,99)),
                mean=float(np.mean(samples)), minimum=min(samples), maximum=max(samples))

def bench_mode(model, x, cpu, mode, iterations):
    import torch
    import numpy as np
    ring, use_graph = mode in ("E2","E3"), mode in ("E1","E3")
    select(model,ring); model.eval()
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc=torch.cuda.memory_allocated()/2**20
    setup_start=time.perf_counter()
    graph=None
    with torch.inference_mode():
        reference=model(x).clone()
        if use_graph:
            graph=InferenceGraph(model,x)
            fn=graph.replay
            first=graph.owned(x)
            first_saved=first.clone()
            changed=x.flip(-1).contiguous(memory_format=torch.channels_last)
            expected=model(changed)
            second=graph.owned(changed)
            torch.testing.assert_close(first,reference,atol=2e-4,rtol=2e-4)
            torch.testing.assert_close(second,expected,atol=2e-4,rtol=2e-4)
            graph.owned(x)
            # The first output must remain unchanged after the second replay.
            torch.testing.assert_close(first,first_saved,atol=0,rtol=0)
            del first,first_saved,second,changed,expected
        else:
            fn=lambda:model(x)
        for _ in range(20): output=fn()
        torch.cuda.synchronize()
        setup_s=time.perf_counter()-setup_start
        setup_peak=torch.cuda.max_memory_allocated()/2**20
        resident=torch.cuda.memory_allocated()/2**20
        pairs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(iterations)]
        for start,end in pairs: start.record();end.record()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        wall,event=[],[]
        for start,end in pairs:
            t=time.perf_counter();start.record();output=fn();end.record();torch.cuda.synchronize()
            wall.append((time.perf_counter()-t)*1000);event.append(start.elapsed_time(end))
        torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(iterations): output=fn()
        torch.cuda.synchronize();throughput_wall=time.perf_counter()-t
        replay_peak=torch.cuda.max_memory_allocated()/2**20
        reserved=torch.cuda.max_memory_reserved()/2**20
        assert torch.isfinite(output).all()
        # Symmetric GPU-input -> independent GPU-output interface timing.
        owned=[]
        for _ in range(20):
            t=time.perf_counter()
            own=graph.owned(x) if use_graph else model(x)
            torch.cuda.synchronize();owned.append((time.perf_counter()-t)*1000)
        # Decoded/normalized pinned CPU batch -> independent CPU logits.
        # JPEG decoding is explicitly outside this measurement for both modes.
        host=[]
        for _ in range(5):
            t=time.perf_counter()
            if use_graph:
                graph.x.copy_(cpu,non_blocking=True);graph.graph.replay();result=graph.output.cpu()
            else:
                gpu=cpu.cuda(non_blocking=True);result=model(gpu).cpu()
            torch.cuda.synchronize();host.append((time.perf_counter()-t)*1000)
        result=dict(mode=mode,wall_ms=wall,event_ms=event,wall=stats(wall),event=stats(event),
                    throughput_images_s=len(x)*iterations/throughput_wall,throughput_wall_s=throughput_wall,
                    gpu_owned_output_ms=owned,host_to_logits_ms=host,setup_s=setup_s,
                    base_allocated_mib=base_alloc,resident_allocated_mib=resident,
                    setup_peak_allocated_mib=setup_peak,replay_peak_allocated_mib=replay_peak,
                    peak_reserved_mib=reserved,graph_different_inputs_and_owned_outputs_verified=use_graph)
        del fn,graph,reference,output,own
    gc.collect();torch.cuda.empty_cache()
    return result

def training_step(model,optimizer,x,target,config):
    import torch
    import torch.nn.functional as F
    optimizer.zero_grad(set_to_none=True)
    logits=model(x)
    loss=F.binary_cross_entropy_with_logits(logits,target)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config["grad_clip_norm"],error_if_nonfinite=True)
    optimizer.step()
    return logits,loss,norm
