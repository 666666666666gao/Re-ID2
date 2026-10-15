"""One selected-recipe seed43..46 wave: parity, native3 pair, full50 pair, GT audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from gpu_thermal_execute import execute
from launch_full_official_frozen_anchor_trial import save
from rgbnt100_expert_seed_contracts import verify_detail, verify_model_record

CONTROLS=(('frequency_shared',2),('axis_shared',3))


def main():
    parser=argparse.ArgumentParser()
    for key in ('selection-file','data-root','pretrained','anchor-root','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--seed',type=int,choices=(43,44,45,46),required=True)
    args=parser.parse_args()
    selection_path=Path(args.selection_file);selection=json.loads(selection_path.read_text())
    assert selection['status']=='ACTUAL_CLOSED_RECIPE_LOCKED_FOR_BOUNDED_EXPERT_N5'
    assert selection['expert_seeds']==[42,43,44,45,46] and selection['teacher_seed']==42
    key=selection['selected_recipe'];assert key in ('AV','BC')
    root=Path(args.output)
    assert root==Path(selection['remote_family_root'])/('s'+str(args.seed))/'RGBNT100'
    root.mkdir(parents=True,exist_ok=False)
    for phase in ('parity','native','training','audit'):(root/phase).mkdir()
    save(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),arguments=vars(args),
        recipe=key,seed=args.seed,teacher_seed=42,physical_gpus=[2,3],maximum_NN=2,
        selection_file_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),temperature_power_control=False,
        scope='One paired normal expert seed only; no automatic next wave or missing/state inference.'))
    anchor=str(Path(args.anchor_root)/'RGBNT100_demo_s42')

    def parity(pair):
        variant,gpu=pair;name='RGBNT100_n5_'+key+'_'+variant+'_seed42_parity'
        out=root/'parity'/name
        exit_row=execute([sys.executable,'-u','check_rgbnt100_seed_factory_parity.py','--expert-recipe',key,
            '--variant',variant,'--data-root',args.data_root,'--pretrained',args.pretrained,
            '--anchor-run-dir',anchor,'--output',str(out)],root/'parity',name,gpu)
        data=json.loads((out/'result.json').read_text())
        assert data['status']=='ACTUAL_SEED42_NATIVE_CONSTRUCTOR_RNG_FOUR_STATE_PARITY_PASS'
        assert data['state_tensors_exact'] and data['CPU_CUDA_RNG_exact'] and data['four_states_exact']
        assert data['optimizer_updates']==data['checkpoint_writes']==0 and data['forward_record_executions']==16
        return dict(result=data,exit=exit_row)

    if args.seed==43:
        with ThreadPoolExecutor(max_workers=2) as pool:parities=list(pool.map(parity,CONTROLS))
    else:
        first=Path(selection['remote_family_root'])/'s43/RGBNT100/native_acceptance.json'
        previous=json.loads(first.read_text())
        assert previous['status']=='PASS' and previous['recipe']==key and previous['seed']==43
        assert previous['selection_file_sha256']==hashlib.sha256(selection_path.read_bytes()).hexdigest()
        parities=previous['parity']
    parity_records=32 if args.seed==43 else 0

    def run(pair,mode):
        variant,gpu=pair
        name='RGBNT100_n5_'+key+'_'+variant+'_s'+str(args.seed)+('_smoke' if mode=='smoke' else '')
        phase='native' if mode=='smoke' else 'training';out=root/phase/name
        exit_row=execute([sys.executable,'-u','run_rgbnt100_expert_N5.py','--expert-recipe',key,
            '--dataset','RGBNT100','--variant',variant,'--seed',str(args.seed),'--mode',mode,
            '--freeze-identity-encoder','1','--data-root',args.data_root,'--pretrained',args.pretrained,
            '--anchor-run-dir',anchor,'--output',str(out)],root/phase,name,gpu)
        return name,out,exit_row

    def native(pair):
        name,out,exit_row=run(pair,'smoke');d=json.loads((out/'smoke.json').read_text())
        assert d['status']=='SMOKE_PASS' and d['steps']==d['attempts']==3 and d['amp_skipped_steps']==0
        assert d['strict_reload_equal'] and all(d['gradients'].values()) and not list(out.glob('*.pth'))
        verify_model_record(d,key,args.seed)
        for detail in d['details']:verify_detail(detail,key,True)
        return d

    with ThreadPoolExecutor(max_workers=2) as pool:natives=list(pool.map(native,CONTROLS))
    assert [(d['names'],d['partial_set']) for d in natives[0]['details']]==[(d['names'],d['partial_set']) for d in natives[1]['details']]
    save(root/'native_acceptance.json',dict(status='PASS',recipe=key,seed=args.seed,controls=2,
        actual_updates=6,paired_sampling=True,parity=parities,parity_model_forward_batches=parity_records//8,
        parity_forward_record_executions=parity_records,parity_reused_from_seed43=args.seed!=43,
        selection_file_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        strict_reload=True,all_active_gradients=True))
    print('RGBNT100_N5_NATIVE_PASS',key,args.seed,flush=True)

    def train(pair):
        name,out,exit_row=run(pair,'train');d=json.loads((out/'result.json').read_text())
        assert d['status']=='COMPLETE' and d['epochs']==50 and d['steps']==d['optimizer_steps']>0
        assert d['amp_skipped_steps']==d['training_heldout_identities']==0
        assert d['training_coverage']==dict(eligible=8675,visited=8675,unvisited=[])
        verify_model_record(d,key,args.seed)
        rows=[json.loads(line) for line in (out/'batch_orders.jsonl').read_text().splitlines()]
        assert len(rows)==d['steps']
        for detail in rows:verify_detail(detail,key,False)
        audited=execute([sys.executable,'-u','audit_full_official_normal.py','--run-dir',str(out)],root/'audit',name,pair[1])
        assert json.loads((out/'normal_cpu_audit.json').read_text())['status']=='PASS'
        files={}
        for file_key,filename in (('raw','best_official_arrays.npz'),('best','best.pth')):
            f=out/filename;files[file_key]=dict(path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        return dict(name=name,variant=pair[0],gpu=pair[1],train=exit_row,audit=audited,
            successful_updates=d['steps'],selected_epoch=d['best']['epoch'],full_metrics=d['full_metrics'],**files)

    with ThreadPoolExecutor(max_workers=2) as pool:trained=list(pool.map(train,CONTROLS))
    orders=[]
    for row in trained:
        values=[json.loads(line) for line in (root/'training'/row['name']/'batch_orders.jsonl').read_text().splitlines()]
        orders.append([(r['epoch'],r['step'],r['names'],r['partial_set']) for r in values])
    assert orders[0]==orders[1]
    assert trained[0]['successful_updates']==trained[1]['successful_updates']
    save(root/'controller_result.json',dict(status='COMPLETE',recipe=key,seed=args.seed,teacher_seed=42,
        controls=2,runs=trained,native_updates=6,successful_updates=sum(r['successful_updates'] for r in trained),
        additional_epochs=100,paired_sampling_exact=True,physical_gpus=[2,3],maximum_NN=2,
        full_training_coverage=True,normal_GT_audited=True,state_cases=0,missing_evaluated=False,
        raw_preserved_pending_local_archive=True,parity_forward_record_executions=parity_records,
        limits='One of four new waves after reusing seed42. Actual per-seed sampler updates, not seed42 order/count. Full50 normal, benchmarkmAP earliesttie; no selected final N5 seed or new whole-pipeline repeats.'))
    print('RGBNT100_N5_WAVE_COMPLETE',key,args.seed,flush=True)


if __name__=='__main__':main()
