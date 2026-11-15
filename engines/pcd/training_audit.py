from collections import defaultdict
from pathlib import Path
import csv,gc,hashlib,json,math
from stage_matrix_common import read,write,digest

def effective_rows(folder,final_step):
    rows={};observed=0
    for session in sorted((folder/'sessions').iterdir()):
        path=session/'train.csv'
        if not path.exists():continue
        start=read(session/'resume_cursor.json')
        assert start is not None
        for step in [s for s in rows if s>start['step']]:del rows[step]
        with path.open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                step=int(row['step']);observed+=1
                if step<=final_step:rows[step]=row
    assert sorted(rows)==list(range(1,final_step+1)),folder
    return rows,observed-len(rows)

def audit_training(root,groups,target,out):
    import torch
    from comparison_support import build_variant,tensor_hash
    from coco_stream import EpochBatches
    from full_worker import configure_torch,learning_rate,load_checkpoint
    from holdout_training_plan import load_plan,planned_lr
    out.mkdir(parents=True,exist_ok=True)
    results=[];batch_hashes=[];cohort_specs=defaultdict(list)
    for group in groups:
        run=Path(group['run']);folder=run/group['variant'];config=read(run/'config.json')
        configure_torch(config)
        index=read(run/'data/train2017.json')
        if config.get('test_train_limit'):index=index[:config['test_train_limit']]
        count=len(index);batches=math.ceil(count/config['batch_size']);final_step=target*batches
        metrics=[read(p) for p in sorted((folder/'epochs').glob('*/metrics.json'))]
        assert [m['epoch'] for m in metrics]==list(range(1,target+1))
        val_count=config.get('test_val_limit',5000)
        assert all(m['train_samples']==count and m['samples']==val_count for m in metrics)
        best=max(metrics,key=lambda m:m['map'])
        payload=load_checkpoint(folder,recover=True)
        assert (payload['epoch'],payload['next_batch'],payload['step'])==(target,0,final_step)
        assert payload['config']==config and 'optimizer' in payload and 'rng' in payload
        spec=read(folder/'model_spec.json');cohort_specs[group['cohort']].append(spec)
        assert payload['dataset_fingerprint']==spec['dataset_fingerprint']
        model,initial=build_variant(group['variant'],config)
        assert initial['state_dict_initial_sha256']==spec['state_dict_initial_sha256']
        initial_a=tensor_hash((n,t) for n,t in model.state_dict().items() if n.endswith(('.A0','.A1','.A2')))
        model.load_state_dict(payload['state_dict'],strict=True)
        assert all(torch.isfinite(t).all() for t in model.state_dict().values())
        assert initial_a==tensor_hash((n,t) for n,t in model.state_dict().items() if n.endswith(('.A0','.A1','.A2')))
        del payload
        best_payload=torch.load(folder/'best_weights.pt',map_location='cpu',weights_only=True,mmap=True)
        assert best_payload['epoch']==best['epoch'] and abs(best_payload['map']-best['map'])<1e-12
        assert best_payload['dataset_fingerprint']==spec['dataset_fingerprint']
        model.load_state_dict(best_payload['state_dict'],strict=True)
        assert all(torch.isfinite(t).all() for t in model.state_dict().values())
        del model,best_payload
        logs,retried=effective_rows(folder,final_step)
        h=hashlib.sha256();plan=load_plan(config)
        for epoch in range(target):
            expected=[index[i][0] for i in EpochBatches(count,config['batch_size'],config['data_seed'],epoch).indices().tolist()]
            loss_sum=wall_sum=0.;images=0
            for batch in range(batches):
                step=epoch*batches+batch+1;r=logs[step];ids=json.loads(r['image_ids'])
                assert ids==expected[batch*config['batch_size']:(batch+1)*config['batch_size']]
                assert (int(r['epoch']),int(r['batch']),int(r['batch_size']))==(epoch+1,batch+1,len(ids))
                assert abs(float(r['lr'])-planned_lr(config,plan,step-1,batches,learning_rate))<1e-14
                assert all(math.isfinite(float(r[k])) for k in ('loss','grad_norm_before_clip','scan_core_grad_norm_before_clip','step_wall_ms'))
                h.update(json.dumps([step,ids],separators=(',',':')).encode())
                images+=len(ids);loss_sum+=float(r['loss'])*len(ids);wall_sum+=float(r['step_wall_ms'])/1000
            assert images==count
            assert abs(loss_sum/count-metrics[epoch]['train_loss'])<1e-10
            assert abs(wall_sum-metrics[epoch]['train_wall_s'])<1e-6
        batch_hashes.append(h.hexdigest())
        incomplete=[];resource_errors=[]
        for session in (folder/'sessions').iterdir():
            summary=read(session/'resource_summary.json')
            if summary is None:incomplete.append(str(session));continue
            resource_errors.extend(error for phase in summary.values() for error in phase.get('errors',[]))
            assert (session/'resources.csv').exists()
            for f in session.glob('gradients_*.json'):
                assert all(not p['missing'] and p['grad_finite'] for p in read(f)['parameters'])
        assert not resource_errors,resource_errors
        results.append({'group':group['id'],'epochs':target,'steps':final_step,'samples_per_epoch':count,
                        'training_order_sha256':h.hexdigest(),'best_epoch':best['epoch'],'best_map':best['map'],
                        'best_weights_sha256':digest(folder/'best_weights.pt'),'strict_reload':True,
                        'resume_state_present':True,'fixed_A_preserved':True,'retry_rows_ignored':retried,
                        'interrupted_sessions_without_resource_summary':incomplete})
        write(out/'audit.json',{'status':'running','groups':results})
        print(f"TRAINING AUDIT PASS {group['id']} / {target} epochs",flush=True)
        del logs;gc.collect()
    assert len(set(batch_hashes))==1
    for specs in cohort_specs.values():
        assert len({s['shared_outer_sha256'] for s in specs})==1
        if all('matched_steps_C_D_sha256' in s for s in specs):assert len({s['matched_steps_C_D_sha256'] for s in specs})==1
    value={'status':'passed','groups':results,'all_group_training_batches_identical':True,'test_data_used':False}
    write(out/'audit.json',value)
    return value
