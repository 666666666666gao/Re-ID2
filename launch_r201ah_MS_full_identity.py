"""Complete-reference parity, new native3, then two full-normal50 AH controls on GPU23."""
import argparse,hashlib,json,os,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from gpu_thermal_execute import execute
from launch_full_official_frozen_anchor_trial import save

CONTROLS=(('frequency_shared',2),('axis_shared',3))


def main():
    parser=argparse.ArgumentParser()
    for key in ('data-root','pretrained','anchor-root','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    for phase in ('reference','native','training','audit','states'):(root/phase).mkdir()
    save(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),arguments=vars(args),physical_gpus=[2,3],maximum_NN=2,
        temperature_power_control=False,experiment='AH full strong5632 reference retained; K private relation corrections'))
    def common(variant,out):
        return ['--variant',variant,'--data-root',args.data_root,'--pretrained',args.pretrained,
            '--anchor-run-dir',str(Path(args.anchor_root)/'MSVR310_demo_s42'),'--output',str(out)]
    def reference(pair):
        variant,gpu=pair;name='MSVR310_r201ah_'+variant+'_reference';out=root/'reference'/name
        row=execute([sys.executable,'-u','evaluate_msvr_full_identity_reference.py',*common(variant,out)],root/'reference',name,gpu)
        d=json.loads((out/'result.json').read_text());assert d['status']=='COMPLETE_AH_FULL5632_FROZEN00_OFFICIAL_TEACHER_PARITY'
        assert d['optimizer_updates']==d['new_weights']==0 and d['strict_reload_equal'] and d['per_query_equal']
        assert (d['train_records'],d['query_records'],d['gallery_records'],d['descriptor_dim'])==(1032,591,1055,5632)
        return dict(name=name,gpu=gpu,exit=row,metrics=d['metrics'])
    with ThreadPoolExecutor(max_workers=2) as pool:references=list(pool.map(reference,CONTROLS))
    assert references[0]['metrics']==references[1]['metrics']
    save(root/'reference_acceptance.json',dict(status='PASS',controls=2,full_frozen00_teacher_GT_per_query_equal=True,metrics=references[0]['metrics']))
    def run(pair,mode):
        variant,gpu=pair;name='MSVR310_r201ah_'+variant+'_s42'+('_smoke' if mode=='smoke' else '')
        phase='native' if mode=='smoke' else 'training';out=root/phase/name
        row=execute([sys.executable,'-u','run_r201ah_MS_full_identity.py','--dataset','MSVR310',
            '--freeze-identity-encoder','1','--seed','42','--mode',mode,*common(variant,out)],root/phase,name,gpu)
        return name,out,row
    def verify_detail(d,native):
        assert d['optimizer_updated'] and d['loss']==d['full_loss'] and d['partial_loss_effective']==0
        assert d['partial_CE_weight']==d['partial_triplet_weight']==0
        assert d['sampling_identities']==16 and d['sampling_instances']==4 and d['sampling_batch']==64
        assert d['primary_full_metric']=='unit5632_margin06_batch_hard_triplet' and d['primary_margin']==.6
        assert d['M_F_auxiliary_weight']==.001 and d['parent_main_CE_weight']==.05 and d['parent_contribution_regression_weight']==0
        assert d['M_F_auxiliary_outlet']=='actual_first11_projected_unit_relation_mean512' and d['auxiliary_original_soft_triplet']
        assert d['descriptor_dim']==d['main_loss_deployed_dimension']==5632 and d['private_bound_dimension']==5120
        assert d['retained_public_dimension']==512 and d['retained_public_projection_frozen']
        c=d['projected_auxiliary_forward_contract'];assert c['projection_calls']==dict(M=3 if native else 2,F=3 if native else 2)
        assert c['pool_calls']==dict(M=2,F=2) and c['neck_calls']==dict(M=1,F=1) and c['feature_shape']==[64,512]
        assert d['final_descriptor_bound_full']['bounded_shift_max']<=.10+1e-6
        if native:
            assert d['fused_BN_calls']==2
            assert all(v['finite'] and v['weighted_auxiliary_l1']>0 for v in d['direct_PM_PF_auxiliary_gradient'].values())
            assert d['relation_PI_native']['per_relation_outside_slot_errors']==[0.]*7
            assert d['relation_PI_native']['state_10_parent_equal'] and d['relation_PI_native']['state_01_parent_equal']
    def native(pair):
        name,out,row=run(pair,'smoke');d=json.loads((out/'smoke.json').read_text())
        assert d['status']=='SMOKE_PASS' and d['steps']==d['attempts']==3 and d['amp_skipped_steps']==0
        assert d['strict_reload_equal'] and d['descriptor_shape']==[8,5632] and all(d['gradients'].values())
        assert not list(out.glob('*.pth')) and (d['parameters'],d['trainable_parameters'])==(100902022,2674822)
        assert d['anchor']['complete_reference']['public_projection_frozen']
        for detail in d['details']:verify_detail(detail,True)
        return d
    with ThreadPoolExecutor(max_workers=2) as pool:natives=list(pool.map(native,CONTROLS))
    assert [(d['names'],d['partial_set']) for d in natives[0]['details']]==[(d['names'],d['partial_set']) for d in natives[1]['details']]
    save(root/'native_acceptance.json',dict(status='PASS',controls=2,actual_updates=6,paired_sampling=True,
        strict_reload_equal=True,all_active_gradients_positive=True,parameters=100902022,trainable_parameters=2674822,descriptor_dim=5632))
    print('AH_NATIVE_PASS',flush=True)
    def train(pair):
        variant,gpu=pair;name,out,row=run(pair,'train');d=json.loads((out/'result.json').read_text())
        assert d['status']=='COMPLETE' and d['epochs']==50 and d['steps']==d['optimizer_steps']==705
        assert d['amp_skipped_steps']==d['training_heldout_identities']==0
        assert d['training_coverage']==dict(eligible=1032,visited=1032,unvisited=[])
        assert (d['parameters'],d['trainable_parameters'],d['descriptor_dim'])==(100902022,2674822,5632)
        orders=[json.loads(line) for line in (out/'batch_orders.jsonl').read_text().splitlines()];assert len(orders)==705
        for detail in orders:verify_detail(detail,False)
        prior=Path('runs/r201ae_MS_strong_staged_20261012/MSVR310/training/MSVR310_r201ae_axis_frozen_s42/batch_orders.jsonl')
        previous=[json.loads(line) for line in prior.read_text().splitlines()]
        identity=lambda values:[(r['epoch'],r['step'],r['names'],r['partial_set']) for r in values]
        assert identity(orders)==identity(previous)
        audited=execute([sys.executable,'-u','audit_msvr_full_identity_normal.py','--run-dir',str(out)],root/'audit',name,gpu)
        assert json.loads((out/'normal_cpu_audit.json').read_text())['status']=='PASS'
        state_dir=root/'states'/name
        state_row=execute([sys.executable,'-u','evaluate_r201ah_MS_normal_states.py','--run-dir',str(out),'--output',str(state_dir)],root/'states',name,gpu)
        state_result=json.loads((state_dir/'result.json').read_text())
        assert state_result['status']=='COMPLETE' and state_result['state_cases']==4
        assert state_result['full11_features_distance_exact_selected_normal'] and state_result['frozen00_full_teacher_six_per_query_equal']
        path=out/'best_official_arrays.npz'
        states_path=state_dir/'normal_states.npz'
        return dict(name=name,gpu=gpu,train=row,audit=audited,states_exit=state_row,
            raw=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            states_raw=dict(path=str(states_path),bytes=states_path.stat().st_size,sha256=hashlib.sha256(states_path.read_bytes()).hexdigest()))
    with ThreadPoolExecutor(max_workers=2) as pool:trained=list(pool.map(train,CONTROLS))
    save(root/'controller_result.json',dict(status='COMPLETE',dataset='MSVR310',runs=trained,references=references,controls=2,
        native_updates=6,successful_updates=1410,additional_epochs=100,state_cases=8,paired_sampling_exact=True,AE_frozen_training_record_order_exact=True,
        physical_gpus=[2,3],maximum_NN=2,raw_preserved_pending_local_archive=True,missing_evaluated=False,new49=False,
        limits='Full strong teacher frozen00 parity precedes newnative3/formal50. Strongteacher gains inherited; ordinary/axis matched within AH, AE differs deployedhead/publicfreeze/interface. Teacher50+extra50 and benchmarkselection disclosed; fullnormal only.'))
    print('AH_CONTROLLER_COMPLETE',flush=True)


if __name__=='__main__':main()
