"""Replay all49 archived conditions through installed-GT CPU checks, two raw files at a time."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, OPTIONS, command, remote_python

pf = PROJECT / 'results/preflight'
review = json.loads((pf / 'full_official_condition_audit_review_20261005.json').read_text(encoding='utf-8'))
assert review['status'] == 'PASS' and not review['blocking_findings']
for file, digest in review['sources_sha256'].items():
    assert hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest
archive = json.loads((pf / 'full_official_public_outlet_identity_distances_local_archive_20261005.json').read_text(encoding='utf-8'))
assert archive['status'] == 'ALL343_PUBLIC_IDENTITY_DISTANCE_FILES_LOCAL_SHA256_VERIFIED_SERVER_COPIES_CLEARED'
name = 'MSVR310_public_identity_axis_shared_s42'
local = Path(archive['local_root'])
source = archive['source']
expected = archive['files'][name]
assert len(expected) == 98
for relative, value in expected.items():
    path = local / relative
    assert path.stat().st_size == value['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256']
validation = '/data/gaob/Re-ID/DeMo-DualAxis/runs/full_official_condition_cpu_equivalence_20261005'
proof_path = pf / 'full_official_condition_archive_equivalence_20261005.json'
assert not proof_path.exists()
command(['scp', *OPTIONS, str(PROJECT / 'audit_full_official_condition.py'), '2026:/data/gaob/Re-ID/DeMo-DualAxis/audit_full_official_condition.py'])
worker = f'''import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,'/data/gaob/Re-ID/DeMo-DualAxis')
from audit_full_official_condition import installed_context,audit_condition
from audit_full_official_control_states import verify_heads
source=Path({source!r});name={name!r};root=Path({validation!r})
assert root.parent==source.parent and not root.exists()
root.mkdir();previous=root/'frozen49';diagnosis=root/'diagnosis'
previous.mkdir();diagnosis.mkdir()
old_previous=source/'frozen49'/name;old_diagnosis=source/'diagnosis'/name
for filename in ('result.json','independent_cpu_audit.json'):
 shutil.copyfile(old_previous/filename,previous/filename)
 shutil.copyfile(old_diagnosis/filename,diagnosis/filename)
for path in old_previous.glob('q_*_g_*.csv'):shutil.copyfile(path,previous/path.name)
for path in old_diagnosis.glob('heads_*'):
 assert path.suffix in ('.csv','.npz');shutil.copyfile(path,diagnosis/path.name)
trained,dataset,installed=installed_context(source/'training'/name)
reported=json.loads((diagnosis/'result.json').read_text())
prior=json.loads((previous/'result.json').read_text())
existing=json.loads((diagnosis/'independent_cpu_audit.json').read_text())
assert reported['status']==prior['status']=='COMPLETE'
assert existing['status']=='PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT'
assert reported['model_arguments']==prior['model_arguments']==trained['arguments']
assert reported['selected_epoch']==prior['selected_epoch']==trained['best']['epoch']
conditions=sorted(reported['measurements']);assert len(conditions)==49
heads,predictions=verify_heads(diagnosis,installed['query'],installed['gallery'],trained['arguments']['gate_gradient_mode'])
assert heads==existing['heads']
receipts={{}};peak_bytes=0
for condition in conditions:
 folder=diagnosis/condition;folder.mkdir()
 for path in (old_diagnosis/condition).glob('*.csv'):shutil.copyfile(path,folder/path.name)
 print(json.dumps(dict(event='READY',condition=condition)),flush=True)
 expected=json.loads(sys.stdin.readline());assert expected['condition']==condition
 paths={{'frozen':previous/(condition+'.npz'),'raw':folder/'raw.npz'}}
 observed={{key:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for key,path in paths.items()}}
 assert observed==expected['files']
 peak_bytes=max(peak_bytes,sum(v['bytes'] for v in observed.values()))
 values,error=audit_condition(folder,previous,reported['measurements'][condition],reported['contributions'][condition],dataset,installed,predictions[condition.split('_')[1]])
 assert values==existing['conditions'][condition]
 receipts[condition]=dict(measured=values,max_sixmetric_error_pp=error,inputs=observed)
 (folder/'condition_cpu_audit.json').write_text(json.dumps(receipts[condition],indent=2)+'\\n')
 assert all(hashlib.sha256(path.read_bytes()).hexdigest()==observed[key]['sha256'] for key,path in paths.items())
 for path in paths.values():
  assert path.resolve().is_relative_to(root);path.unlink()
 assert not list(root.rglob('raw.npz')) and not list(previous.glob('q_*_g_*.npz'))
 print(json.dumps(dict(event='PASS',condition=condition,receipt=receipts[condition])),flush=True)
assert len(receipts)==49
result=dict(status='ALL49_FULL_OFFICIAL_CONDITION_CPU_AUDITS_EXACT_WITH_BOUNDED_RAW_REPLAY',conditions=receipts,cases=294,full_split_counts={{s:len(rows) for s,rows in installed.items()}},max_live_raw_files=2,peak_live_raw_bytes=peak_bytes,neural_execution=0,optimizer_updates=0,training_heldout_identities=0)
(root/'result.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(dict(event='COMPLETE',result=result)),flush=True)
'''
process = subprocess.Popen(['ssh', *OPTIONS, '2026', '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python -u -c ' + shlex.quote(worker)],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8')
receipts = {}
result = None
for line in process.stdout:
    message = json.loads(line)
    condition = message.get('condition')
    if message['event'] == 'READY':
        selected = {'frozen': 'frozen49/' + name + '/' + condition + '.npz',
                    'raw': 'diagnosis/' + name + '/' + condition + '/raw.npz'}
        targets = {'frozen': validation + '/frozen49/' + condition + '.npz',
                   'raw': validation + '/diagnosis/' + condition + '/raw.npz'}
        for key, relative in selected.items():
            command(['scp', *OPTIONS, str(local / relative), '2026:' + targets[key]])
        process.stdin.write(json.dumps(dict(condition=condition, files={k: expected[v] for k,v in selected.items()})) + '\n')
        process.stdin.flush()
    elif message['event'] == 'PASS':
        assert condition not in receipts
        receipts[condition] = message['receipt']
        print('FULL_OFFICIAL_CONDITION_CPU_REPLAY_PASS', condition, flush=True)
    else:
        assert message['event'] == 'COMPLETE' and result is None
        result = message['result']
process.stdin.close()
assert process.wait() == 0
assert result is not None and result['conditions'] == receipts and len(receipts) == 49
assert result['cases'] == 294 and result['full_split_counts'] == dict(train=1032, query=591, gallery=1055)
assert result['neural_execution'] == result['optimizer_updates'] == 0
result.update(verified_at=datetime.now().isoformat(timespec='seconds'), remote_validation=validation,
              local_raw_archive=str(local), source_review='results/preflight/full_official_condition_audit_review_20261005.json',
              limits='Existing full MSVR axis distances replayed, not new NN output. Per-condition CPU audit/local restoration/bounded cleanup is validated for all49; future RGBNT100 neural streaming scheduling remains unimplemented and unvalidated.')
proof_path.write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'))
print('ALL49_FULL_OFFICIAL_CONDITION_CPU_REPLAY_VERIFIED',result['peak_live_raw_bytes'],flush=True)
