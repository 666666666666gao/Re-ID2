"""Retire exactly four closed, dominated legacy fit/dev weights, preserving evidence."""
from datetime import datetime
import csv,hashlib,json,shlex,sys
from pathlib import Path

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command


def main():
    pf=PROJECT/'results/preflight'
    review=json.loads((pf/'r201i_expert_seeds_prelaunch_recovery_source_review_20261007.json').read_text(encoding='utf-8'))
    assert review['status']=='PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in review['sources_sha256'].items())
    proof=pf/'closed_legacy_M1_weight_capacity_retirement_actual_20261007.json';assert not proof.exists()
    inventory=json.loads((pf/'closed_legacy_M1_weight_capacity_inventory_20261007.json').read_text(encoding='utf-8'))
    candidates=inventory['records'][:4];retained=inventory['records'][4:]
    assert len(candidates)==4 and len(retained)==2 and not inventory['current_candidate_consumers'] and not inventory['seed_remote_output_exists']
    evidence=[]
    for row in candidates:
        parts=row['path'].split('/');assert parts[:2]==['runs','common_outlet_m1_trial_20261003_v3'] and parts[2] in ('original_mean','seed_query') and parts[3]=='development' and parts[5]=='best.pth'
        model=parts[4];variant='axis_shared' if model=='MSVR310_axis_shared_s42' else 'frequency_shared'
        assert model=='MSVR310_'+variant+'_s42'
        keep=next(r for r in retained if ('MSVR310_'+variant+'_s42') in r['path'])
        assert all(r['status']=='COMPLETE' and r['epochs']==50 and r['query_count']==210 and r['gallery_count']==360 and r['descriptor_dim']==5632 for r in (row,keep))
        assert keep['mAP']>row['mAP'] and keep['Rank1']>=row['Rank1']
        old=PROJECT/'results/common_outlet_m1_v3_complete'/parts[2]/'frozen49'/model
        new=PROJECT/'results/common_coordinate_v12_trial_20261003/frozen49'/model
        assert len(list(old.glob('q_*_g_*.csv')))==len(list(new.glob('q_*_g_*.csv')))==49
        exit_record=json.loads((old.parent/(model+'_exit.json')).read_text(encoding='utf-8'));assert exit_record['exit_code']==0
        paths=[old/'q_RNT_g_RNT.csv',new/'q_RNT_g_RNT.csv']
        tables=[]
        for path,metric in zip(paths,(row,keep)):
            with path.open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
            assert len(rows)==210
            assert abs(sum(float(r['AP']) for r in rows)/210*100-metric['mAP'])<1e-7
            assert abs(sum(int(r['Rank-1']) for r in rows)/210*100-metric['Rank1'])<1e-7
            tables.append(rows)
        keys=('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')
        assert all(all(a[k]==b[k] for k in keys) for a,b in zip(*tables))
        evidence.append(dict(retire=row,retain=keep,all49_text_preserved=True,closed_exit=exit_record,normal_query_GT_metadata_exact=True,CSV_sha256={p.relative_to(PROJECT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    code=f'''from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/DeMo-DualAxis');rows={inventory['records']!r};candidates={candidates!r}
assert root.resolve()==Path('/data/gaob/Re-ID/DeMo-DualAxis') and not (root/'runs/r201i_expert_seeds_20261006').exists()
lines=subprocess.run(['ps','-eo','pid,stat,args'],check=True,capture_output=True,text=True).stdout.splitlines()
assert not any(r['path'].rsplit('/',1)[0] in line for r in candidates for line in lines)
for row in rows:
 p=root/row['path'];assert p.resolve().is_relative_to(root/'runs') and p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 assert hashlib.sha256((p.parent/'result.json').read_bytes()).hexdigest()==row['result_sha256']
before=shutil.disk_usage(root).free
for row in candidates:(root/row['path']).unlink()
assert all(not (root/r['path']).exists() for r in candidates)
assert all((root/r['path']).is_file() and hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in rows[4:])
after=shutil.disk_usage(root).free;assert after>3_500_000_000
print(json.dumps(dict(removed_files=4,removed_bytes=sum(r['bytes'] for r in candidates),remote_free_before=before,remote_free_after=after,retained_two_V12_hashes_exact=True,current_candidate_consumers=0,all_other_weights_untouched=True)))'''
    actual=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')+' -'],input=code))
    result=dict(status='ACTUAL_FOUR_CLOSED_DOMINATED_LEGACY_M1_WEIGHTS_RETIRED_SEED_CAPACITY_RESTORED',completed_at=datetime.now().isoformat(timespec='seconds'),actual=actual,evidence=evidence,new_neural_calls=0,new_optimizer_updates=0,limits='Only these four historical artificial fit/dev checkpoint binaries removed. Same protocol retained V12 controls dominate normal mAP/Rank1; all49 text/GT metadata/source retained. Legacy neural replay requires retraining. Original full-official anchors, G/I selected weights, current/new seeds and all unique raw arrays untouched.')
    proof.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],**actual)),flush=True)


if __name__=='__main__':main()
