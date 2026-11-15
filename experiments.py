"""Prepare/train/resume/evaluate the fourteen experiment groups without changing their engines."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from model_api import engine_directory, ROOT, activate, build_model, configuration, group, groups, inference_graph


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def run_info(args):
    run = args.run.resolve()
    info = read(run / 'experiment.json')
    if info['id'] != args.group:
        raise ValueError('Run directory belongs to another group.')
    return run, group(args.group), read(run / 'config.json')


def prepare(args):
    config = configuration(args.group)
    index = args.index.resolve()
    meta = read(index / 'dataset.json')
    assert meta['splits']['train2017']['count'] == 94627 and meta['splits']['val2017']['count'] == 5000
    config['data_root'] = meta['data_root']
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / 'data').mkdir()
    # The training directory deliberately contains no held-out test index.
    meta['splits'] = {k: meta['splits'][k] for k in ('train2017', 'val2017')}
    meta.pop('fingerprint_sha256', None)
    meta['holdout_test_sealed'] = True
    for name, item in meta['splits'].items():
        src = index / (name + '.json')
        assert hashlib.sha256(src.read_bytes()).hexdigest() == item['index_sha256']
        shutil.copy2(src, run / 'data' / src.name)
    meta['fingerprint_sha256'] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    write(run / 'data/dataset.json', meta)
    plan = dict(initial_lr_horizon_epochs=40, target_epochs=1 if args.smoke else 40,
                extensions=[], test_evaluation_enabled=False)
    if args.smoke:
        config.update(test_train_limit=99, test_val_limit=16, loader_workers=0,
                      progress_every=1, gradient_probe_every=1)
    config['training_plan_file'] = str(run / 'training_plan.json')
    write(run / 'training_plan.json', plan)
    write(run / 'config.json', config)
    write(run / 'experiment.json', dict(id=args.group, index_directory=str(index), smoke_only=args.smoke))
    print('Prepared', run)


def train(args):
    run, info, config = run_info(args)
    pause = run / 'PAUSE'
    if args.resume:
        pause.unlink(missing_ok=True)
    elif (run / info['variant'] / 'checkpoint.pt').exists():
        raise ValueError('Use --resume to continue an existing checkpoint.')
    engine = engine_directory(info['engine'])
    env = os.environ.copy()
    env.update(TORCHDYNAMO_DISABLE='1', WANDB_MODE='disabled', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    env.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / '.torch_extensions' / info['engine']))
    while not pause.exists():
        progress = run / info['variant'] / 'progress.json'
        if progress.exists() and read(progress)['epochs_completed'] >= read(run / 'training_plan.json')['target_epochs']:
            print('Training target completed. Evaluate explicitly when ready.')
            return
        command = [sys.executable, str(engine / 'full_worker.py'), '--run', str(run), '--variant', info['variant']]
        if args.max_steps:
            command += ['--max-steps', str(args.max_steps)]
        result = subprocess.run(command, cwd=engine, env=env)
        if result.returncode == 75:
            print('Paused after saving checkpoint.')
            return
        result.check_returncode()
        if args.max_steps:
            return
    print('Run is paused; use train --resume to continue.')


def evaluate(args):
    import torch
    from torch.nn import functional as F
    run, info, config = run_info(args)
    activate(args.group)
    from coco_stream import CocoImages, make_loader
    from comparison_support import multilabel_metrics
    index = args.index.resolve() if args.index else run / 'data'
    split = 'val2017' if args.split == 'val' else 'test_holdout2017'
    if not (index / (split + '.json')).exists():
        raise ValueError('For test evaluation pass --index with the full frozen data index.')
    checkpoint = args.checkpoint or run / info['variant'] / 'best_weights.pt'
    model, spec = build_model(args.group, checkpoint, config=config)
    model.eval()
    is_smoke = read(run / 'experiment.json').get('smoke_only', False)
    if is_smoke and args.split == 'test':
        raise ValueError('Smoke runs do not evaluate the held-out test set.')
    dataset = CocoImages(index, split, config['resolution'], config['data_seed'],
                         limit=config.get('test_val_limit') if is_smoke else None)
    loader = make_loader(dataset, config)
    outputs, targets, image_ids, captured = [], [], [], {}
    started = time.perf_counter()
    with torch.inference_mode():
        for x, y, ids in loader:
            x = x.cuda().contiguous(memory_format=torch.channels_last)
            if info['engine'] == 'pcd':
                key = tuple(x.shape)
                if key not in captured:
                    captured[key] = inference_graph(model, x)
                result = captured[key].owned(x)
            else:
                result = model(x)
            outputs.append(result.float().cpu()); targets.append(y)
            image_ids.extend(int(i) for i in ids)
    logits, labels = torch.cat(outputs), torch.cat(targets)
    assert torch.isfinite(logits).all()
    metrics = multilabel_metrics(logits, labels)
    metrics.update(bce=F.binary_cross_entropy_with_logits(logits, labels).item(), samples=len(dataset),
                   seconds_including_loading=time.perf_counter()-started)
    output = args.output or run / 'evaluations' / args.split
    output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(logits=logits, labels=labels, image_ids=image_ids), output / 'predictions.pt')
    write(output / 'metrics.json', dict(group=args.group, split=split, metrics=metrics,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        graph=info['engine']=='pcd', smoke_only=is_smoke))
    if args.split == 'test':
        plan = read(run / 'training_plan.json'); plan['test_evaluation_enabled'] = True
        write(run / 'training_plan.json', plan)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def benchmark(args):
    import torch
    import psutil
    run, info, config = run_info(args)
    activate(args.group)
    from coco_stream import CocoImages
    process=psutil.Process()
    affinity=None
    if hasattr(process,'cpu_affinity'):
        available=process.cpu_affinity()
        affinity=[i for i in config['inference']['cpu_affinity'] if i in available]
        if affinity:process.cpu_affinity(affinity)
    model, _ = build_model(args.group, args.checkpoint or run/info['variant']/'best_weights.pt', config=config)
    model.eval()
    index = args.index or run/'data'
    data = CocoImages(index, 'val2017', args.resolution, config['data_seed'])
    samples = [data[i] for i in range(args.batch)]
    cpu = torch.stack([x[0] for x in samples]).contiguous(memory_format=torch.channels_last).pin_memory()
    output = args.output or run/'benchmarks'/f'{args.resolution}-b{args.batch}'
    output.mkdir(parents=True, exist_ok=False)
    records=[]
    for repeat in range(args.rounds):
        if info['engine'] == 'pcd':
            from measurement_primitives import bench_mode
            x=cpu.cuda()
            row=bench_mode(model, x, cpu, 'E1', args.iterations)
        else:
            from comparison_benchmark import measure
            paths=[data.directory/data.rows[i][1] for i in range(args.batch)]
            row=measure(model,cpu,paths,'fp32',dict(warmup=20,iterations=args.iterations,throughput_iterations=args.iterations))
        row['round']=repeat;records.append(row)
    write(output/'timings.json',dict(group=args.group,precision='fp32',batch=args.batch,resolution=args.resolution,
        graph=info['engine']=='pcd',rounds=records,cpu_affinity=affinity,
        scope='Historical measurement bodies; resource profiling is separate'))
    # Sampling and profiler passes never overlap the primary timing samples.
    from resource_monitor import ResourceMonitor
    x=cpu.cuda();graph=None
    with torch.inference_mode():
        if info['engine']=='pcd': graph=inference_graph(model,x)
        def forward():
            if graph:
                graph.graph.replay();return graph.output
            return model(x)
        with ResourceMonitor(output/'resources.csv',config['telemetry_interval_s']) as monitor:
            monitor.set_phase('inference_profile')
            for _ in range(100): forward()
            torch.cuda.synchronize();monitor.set_phase('complete')
        write(output/'resources.json',monitor.summary('inference_profile'))
        if args.trace:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as profiler:
                for _ in range(5): forward()
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(output/'cuda_trace.json'))
    print('Benchmark saved:',output)


def smoke(args):
    import torch
    model, spec=build_model(args.group)
    model.eval()
    with torch.inference_mode():
        x=torch.randn(1,3,224,224,device='cuda').contiguous(memory_format=torch.channels_last)
        eager=model(x)
        assert eager.shape==(1,80) and torch.isfinite(eager).all()
        if group(args.group)['engine']=='pcd':
            graph=inference_graph(model,x)
            torch.testing.assert_close(graph.owned(x),eager,atol=2e-4,rtol=2e-4)
    result=dict(group=args.group,parameters=spec['parameters'],finite_output=True,graph_checked=group(args.group)['engine']=='pcd')
    if args.output:write(args.output,result)
    print(json.dumps(result))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('list')
    ids=[g['id'] for g in groups()]
    for name in ('prepare','train','pause','status','extend','evaluate','benchmark','smoke'):
        cmd=sub.add_parser(name)
        cmd.add_argument('--group',choices=ids,required=True)
        if name!='smoke':cmd.add_argument('--run',type=Path,required=True)
        if name in ('prepare','evaluate','benchmark'):cmd.add_argument('--index',type=Path,required=name=='prepare')
        if name=='prepare':cmd.add_argument('--smoke',action='store_true')
        if name=='train':
            cmd.add_argument('--resume',action='store_true');cmd.add_argument('--max-steps',type=int,default=0)
        if name=='extend':
            cmd.add_argument('--epochs',type=int,required=True);cmd.add_argument('--lr',type=float,required=True);cmd.add_argument('--min-lr',type=float,default=1e-7)
        if name in ('evaluate','benchmark'):
            cmd.add_argument('--checkpoint',type=Path);cmd.add_argument('--output',type=Path)
        if name=='evaluate':cmd.add_argument('--split',choices=['val','test'],default='val')
        if name=='benchmark':
            cmd.add_argument('--batch',type=int,default=1);cmd.add_argument('--resolution',type=int,default=224)
            cmd.add_argument('--rounds',type=int,default=3);cmd.add_argument('--iterations',type=int,default=100);cmd.add_argument('--trace',action='store_true')
        if name=='smoke':cmd.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.action=='list':print(json.dumps(groups(),ensure_ascii=False,indent=2))
    elif args.action=='prepare':prepare(args)
    elif args.action=='train':train(args)
    elif args.action=='pause':
        run,_,_=run_info(args);(run/'PAUSE').write_text('Requested graceful pause',encoding='utf8')
    elif args.action=='status':
        run,info,_=run_info(args);path=run/info['variant']/'progress.json';print(json.dumps(read(path) if path.exists() else {'status':'not_started'},indent=2))
    elif args.action=='extend':
        run,_,_=run_info(args);plan=read(run/'training_plan.json')
        if plan['test_evaluation_enabled']:raise ValueError('Test has already been evaluated. Use a new experiment for an extended budget.')
        start=plan['extensions'][-1]['end_epoch'] if plan['extensions'] else 40
        assert args.epochs>start and 0<args.min_lr<=args.lr
        plan['extensions'].append(dict(start_epoch=start,end_epoch=args.epochs,lr=args.lr,min_lr=args.min_lr))
        plan['target_epochs']=args.epochs;write(run/'training_plan.json',plan)
    elif args.action=='evaluate':evaluate(args)
    elif args.action=='benchmark':benchmark(args)
    else:smoke(args)


if __name__=='__main__':main()
