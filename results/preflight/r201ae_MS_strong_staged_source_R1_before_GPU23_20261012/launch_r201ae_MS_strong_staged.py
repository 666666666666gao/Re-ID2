"""Four AE controls; native admission, full50, installed GT, four states, preserved raw."""
import argparse,hashlib,json,os,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from gpu_01_diagnostic_execute import execute as execute01
from gpu_thermal_execute import execute as execute23
from launch_full_official_frozen_anchor_trial import save

CONTROLS=(('axis_frozen',0),('axis_staged',1),('frequency_staged',2),('demo_staged',3))
LOCK=threading.Lock()


def emit(row):
    with LOCK:
        print('R201AE_STREAM '+json.dumps(row),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for key in ('data-root','pretrained','anchor-root','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    for key in ('native','training','audit','states'):(root/key).mkdir()
    save(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),arguments=vars(args),
        controls=CONTROLS,maximum_NN=4,temperature_power_control=False,scope='Strong private MS normal only; fixed outlets5/relations45, same-budget noexpert control; all native before formal'))
    def execute(control,gpu,mode):
        name='MSVR310_r201ae_'+control+'_s42'+('_smoke' if mode=='smoke' else '')
        phase='native' if mode=='smoke' else 'training';output=root/phase/name
        runner=execute01 if gpu<2 else execute23
        launch=runner([sys.executable,'-u','run_r201ae_MS_strong_staged.py','--dataset','MSVR310','--variant',control,
            '--freeze-identity-encoder','1','--seed','42','--mode',mode,'--data-root',args.data_root,
            '--pretrained',args.pretrained,'--anchor-run-dir',str(Path(args.anchor_root)/'MSVR310_demo_s42'),'--output',str(output)],root/phase,name,gpu)
        return name,output,launch
    def native(control,gpu):
        name,output,launch=execute(control,gpu,'smoke');data=json.loads((output/'smoke.json').read_text())
        assert data['status']=='SMOKE_PASS' and data['steps']==data['attempts']==3 and data['amp_skipped_steps']==0
        assert data['strict_reload_equal'] and all(data['gradients'].values()) and not list(output.glob('*.pth'))
        assert data['descriptor_dim']==5120 and data['descriptor_shape']==[8,5120]
        assert (data['train_records'],data['query_records'],data['gallery_records'])==(1032,591,1055)
        expected=['outlets']*3 if control=='axis_frozen' else ['outlets','relations','relations']
        assert [row['adaptation_phase'] for row in data['details']]==expected
        for row in data['details']:
            assert row['optimizer_updated'] and row['control']==control and row['fused_BN_calls']==2
            assert row['partial_loss_effective']==row['partial_CE_weight']==row['partial_triplet_weight']==0
            assert row['loss']==row['full_loss'] and row['primary_margin']==.6
            assert row['generalFusion_effective_LR_factor']==(.1 if row['adaptation_phase']=='relations' else 0.)
            assert row['nominal_group_LR_restored'] and row['core_eval_modes_unchanged']
            if row['adaptation_phase']=='relations':assert all(row['generalFusion_native_gradients'].values())
            else:assert not any(row['generalFusion_native_gradients'].values())
            if control!='demo_staged':
                assert row['M_F_auxiliary_weight']==.001 and row['parent_main_CE_weight']==.05 and row['parent_contribution_regression_weight']==0
                assert all(g['finite'] and g['weighted_auxiliary_l1']>0 for g in row['direct_PM_PF_auxiliary_gradient'].values())
            else:assert row['no_new_experts'] and row['M_F_auxiliary_weight']==row['contribution_regression_weight']==0
        assert data['anchor']['strong_teacher']['weight_sha256']=='7744f4f7f796e17e778b2d23e5c025c2197029610634cac26c88ba9f73121fb3'
        return data
    with ThreadPoolExecutor(max_workers=4) as pool:
        native_rows=list(pool.map(lambda pair:native(*pair),CONTROLS))
    assert len({(r['parameters'],r['trainable_parameters_by_phase']['relations']) for r in native_rows[1:3]})==1
    assert all([(d['names'],d['partial_set']) for d in r['details']]==[(d['names'],d['partial_set']) for d in native_rows[0]['details']] for r in native_rows)
    save(root/'native_acceptance.json',dict(status='PASS',controls=4,actual_updates=12,phase_boundary_actual=True,
        paired_sampling=True,all_active_gradients_positive=True,strict_reload_equal=True,parameter_differences_disclosed=True))
    emit(dict(event='NATIVE_PASS',controls=4,actual_updates=12,phase_boundary_actual=True))
    def train(control,gpu):
        name,output,launch=execute(control,gpu,'train');data=json.loads((output/'result.json').read_text())
        assert data['status']=='COMPLETE' and data['epochs']==50 and data['steps']==data['optimizer_steps']==705
        assert data['amp_skipped_steps']==data['training_heldout_identities']==0
        assert data['training_coverage']==dict(eligible=1032,visited=1032,unvisited=[])
        orders=[json.loads(line) for line in (output/'batch_orders.jsonl').read_text().splitlines()]
        assert len(orders)==705 and all(row['optimizer_updated'] and row['control']==control for row in orders)
        assert all(row['adaptation_phase']==('relations' if control!='axis_frozen' and row['epoch']>5 else 'outlets') for row in orders)
        runner=execute01 if gpu<2 else execute23
        audited=runner([sys.executable,'-u','audit_full_official_normal.py','--run-dir',str(output)],root/'audit',name,gpu)
        assert json.loads((output/'normal_cpu_audit.json').read_text())['status']=='PASS'
        states=root/'states'/name
        state_launch=runner([sys.executable,'-u','evaluate_r201ae_MS_normal_states.py','--run-dir',str(output),'--output',str(states)],root/'states',name,gpu)
        state=json.loads((states/'result.json').read_text());assert state['status']=='COMPLETE' and state['full11_six_metrics_equal_selected_normal']
        return dict(name=name,control=control,gpu=gpu,train=launch,audit=audited,states=state_launch)
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda pair:train(*pair),CONTROLS))
    raw={}
    for row in results:
        name=row['name']
        for kind,path in [('normal',root/'training'/name/'best_official_arrays.npz'),('states',root/'states'/name/'normal_states.npz')]:
            raw[name+'/'+kind]=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save(root/'raw_manifest.json',dict(status='REMOTE_RAW_PRESERVED_PENDING_LOCAL_TRANSFER',files=raw))

    def order(control):
        p=root/'training'/('MSVR310_r201ae_'+control+'_s42')/'batch_orders.jsonl'
        return [(r['epoch'],r['step'],r['names'],r['partial_set']) for r in map(json.loads,p.read_text().splitlines())]
    assert all(order(control)==order('axis_frozen') for control,_ in CONTROLS)
    save(root/'controller_result.json',dict(status='COMPLETE',dataset='MSVR310',controls=4,runs=results,
        additional_epochs=200,successful_updates=2820,native_updates=12,paired_sampling_exact=True,
        normal_raws_preserved=4,state_raws_preserved=4,local_transfer_completed=False,missing_evaluated=False,
        limits='Baseline initialization, relation adaptation and expert increments separated. All full official normal/GT/samebest six/fourstates; no missing/49/seeds/201.'))
    emit(dict(event='CONTROLLER_COMPLETE',controls=4,successful_updates=2820))


if __name__=='__main__':main()
