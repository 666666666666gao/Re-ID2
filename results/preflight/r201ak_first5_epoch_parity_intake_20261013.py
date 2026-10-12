"""Read completed warmup CSV rows after the scheduled first-stage observation."""
from pathlib import Path
from datetime import datetime
import csv,hashlib,json,shlex,sys

PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
PF=PROJECT/'results/preflight'
KEYS=('epoch','steps','optimizer_steps','amp_skipped_steps','amp_scale','loss',
      'mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20',
      'epoch_unique_train_records','cumulative_unique_train_records')


def main():
    phase_path=PF/'r201ak_first_phase_actual_20261013.json'
    phase=json.loads(phase_path.read_text())
    admission=json.loads((PF/'r201ak_native_and_formal_launch_actual_20261013.json').read_text())
    assert phase['observed_at']>=admission['first_phase_observation_not_before_epoch_seconds']
    assert phase['controller_pid']==admission['controller_pid'] and phase['controller_alive']
    assert all(row['alive'] and row['latest_batch']['adaptation_phase']=='visual_tail'
        and row['latest_batch']['stage_epoch']>=6 for row in phase['workers'].values())
    code='''import csv,json
from pathlib import Path
r=Path('/data/gaob/Re-ID/DeMo-DualAxis/runs/r201ak_RGB_lastblock_staged_20261013/RGBNT100/training')
data={}
for variant in ('frequency_shared','axis_shared'):
 f=r/('RGBNT100_r201ak_'+variant+'_s42')/'epochs.csv'
 with f.open(newline='') as stream:rows=list(csv.DictReader(stream))
 assert len(rows)>=5 and [int(x['epoch']) for x in rows[:5]]==[1,2,3,4,5]
 data[variant]=rows[:5]
print(json.dumps(data))'''
    actual=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')+' -'],input=code))
    inputs={phase_path.relative_to(PROJECT).as_posix():hashlib.sha256(phase_path.read_bytes()).hexdigest()}
    comparisons=[];differences=[]
    for variant,current in actual.items():
        f=PROJECT/'results/r201ag_RGB_coordinate_metric_20261012/RGBNT100/training'/('RGBNT100_r201ag_'+variant+'_s42')/'epochs.csv'
        with f.open(encoding='utf-8',newline='') as stream:old=list(csv.DictReader(stream))[:5]
        inputs[f.relative_to(PROJECT).as_posix()]=hashlib.sha256(f.read_bytes()).hexdigest()
        for before,after in zip(old,current):
            delta={key:float(after[key])-float(before[key]) for key in KEYS}
            changed={key:value for key,value in delta.items() if value!=0}
            comparisons.append(dict(variant=variant,epoch=int(after['epoch']),AG=before,AK=after,delta=delta,changed_fields=changed))
            if changed:differences.append(dict(variant=variant,epoch=int(after['epoch']),changed=changed))
    output=PF/'r201ak_first5_epoch_parity_actual_20261013.json';assert not output.exists()
    result=dict(status='ACTUAL_FIRST5_COMPLETED_EPOCH_AG_AK_LOG_RECOUNT',
        at=datetime.now().astimezone().isoformat(timespec='seconds'),exact_equal=not differences,
        completed_epoch_rows=10,fields_per_row=len(KEYS),differences=differences,comparisons=comparisons,
        inputs_sha256=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_neural_calls=0,new_optimizer_updates=0,CPU_remote_csv_intakes=1,
        limits='Scheduled first-stage readout of completed epoch logs. Wall seconds excluded; exact numeric log equality is not tensor/weight parity or final performance. Both controllers continue unchanged; this does not assert all50 epochs completed or original teacher equality after adaptation.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','exact_equal','completed_epoch_rows','fields_per_row','differences')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
