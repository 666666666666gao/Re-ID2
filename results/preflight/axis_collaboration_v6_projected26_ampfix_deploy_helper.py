from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python

review=json.loads((PROJECT/'results/preflight/axis_collaboration_v6_projected_ampfix_review.json').read_text(encoding='utf-8'))
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
files=('projected_mass_axis_collaboration.py','run_projected_mass_experiment.py','verify_projected_axis_mass.py','launch_projected_mass_trial.py')
new_sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files}
assert review['checked_source_sha256']==new_sources
original=json.loads((PROJECT/'results/preflight/axis_collaboration_v5_mass_launch.json').read_text(encoding='utf-8'))['source_sha256']
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in original.items())
plan=json.loads((PROJECT/'results/preflight/axis_collaboration_v6_projected_plan.json').read_text(encoding='utf-8'))
assert plan['status']=='AMP_FAILURE_CAPTURED_REPAIR_AWAITING_REAL_PREFLIGHT' and plan['budget_epochs']==50
root,python=HOSTS['2026'];output=root+'/runs/axis_collaboration_v6_projected_mass_trial_ampfix'
failed_sources={'run_experiment.py': '6efe9bc31912e90a2381678941a6b209aaba5949a824d02163fb783ba6869154', 'scaled_axis_collaboration.py': '0db085f891653cdca1b821a04e4bd019ff644edf42a481be57bd818abafb3f30', 'diagnose_axis_collaboration.py': '60f3c6217cdebb347ab13501bb0013ca52925b139ba333cc2f2f6ec90c3e6729', 'launch_scaled_diagnostic.py': 'ca865b0204806a0f391d18fe4d91710a90d006a981d1585958e473488c612a1b', 'launch_axis_scaled.py': 'b0e069299966790340d9aaba55c1318ac18183a6def169d9cc5e9de655b58e9f', 'axis_collaboration.py': '9146c6133f183c3a5f5867c72670168f6b97d31af00e80cf9dd751e2b03c8b69', 'residual_dual_axis.py': '4a22801f014e971786402a477a33e7f86ecf608e786ab4c2f3c796abc033c91c', 'dual_axis.py': '0c6090fe5b8b15cc021bf1809860f0675187288dc2145d0129ce89a5eb78a330', 'experiment_data.py': '70775c156c0ab8d31c89dbc08e5544405ca42dfdb015f4f39149f1d034b7c4d3', 'full_evaluation.py': 'a5fe4bf2927bdc43df80a73098685a90281aaaf68f7d8e838b322508c5957427', 'modeling/make_model.py': '3aa4b3056475e5b2662465d2ef732872882909a8f17a574dece9465e8391a131', 'utils/reid_evaluation.py': '96dd0c93d61b67be1944bdb90d59c64ef2246a0c65c776f1e24c27ca858560f2', 'splits.json': '66f6ee8f1c07f4fb97bf48ed8189b60859ab2f5f207ceea6b28a90c4ee1f9377', 'configs/MSVR310/DeMo.yml': '099085c58f031a9edfb86be60d801e7a9f48e080cb0629c11a63d9d3b7a59b1e', 'missing_development.py': '9cbb3440b61ecf3bddfe88bf071328632c9ad3903b2d02aa0bf5a36d57cf8973', 'mass_axis_collaboration.py': '1312c8b993dcd16b1db1a8845a247afd4b1eff13255832644dd30e029e268f8d', 'run_mass_experiment.py': 'ad3351c5ca83c21b57ce1454c0ad00673396ce5cbcfd0ccfaaa63f7eecb7e07b', 'verify_axis_mass.py': '0d9df0e166d5f7a2260356acde8abcfc673cec630bd2935aab142755e2312dba', 'projected_mass_axis_collaboration.py': 'ae0cbd086495af52ed3461f5cd8719a982b1e8ffcc77ff924a21e6c99455de70', 'run_projected_mass_experiment.py': '4471f8eb6507e5562242765ac6833859a1bf0be208db7b4364cc2215f24ec98e', 'verify_projected_axis_mass.py': '00149ff5bcc4d8d99d5ce85a13ecc1bb6ff4d45f6884154aa358a97eb71e2c2f', 'launch_projected_mass_trial.py': '67a09be049d6ea95347c691950145527b43e2ab4f31b1c11188b2494fd47212c'}
failed_pids=[3100567, 3100699]
guard=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {original!r}.items())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {failed_sources!r}.items())
assert not Path({output!r}).exists()
assert all(not subprocess.run(['ps','-p',str(pid),'-o','pid='],capture_output=True,text=True).stdout.strip() for pid in {failed_pids!r})
used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<500
free=shutil.disk_usage(root).free;assert free>2600000000
print(json.dumps(dict(gpu1_memory_used_MiB=used,disk_free_bytes=free,disk_budget='actual1.197GB smoke+399MB best+399MB terminal(model/noAdam)=1.995GB; reserveup to282MB V5RGB1004states plus323MB buffers. V5three training last files already complete; no existing artifacts deleted',warm_environment='same actual2026 V5 preflight and 50epoch environment, no mutation')))
'''
availability=json.loads(remote_python('2026',guard))
for name in files[:1]:command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+root+'/'+name])
sources={**original,**new_sources}
start=f'''import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<500
assert shutil.disk_usage(root).free>2600000000
output=Path({output!r});assert not output.exists()
log=Path(str(output)+'.log');assert not log.exists()
argv=[{python!r},'-u','launch_projected_mass_trial.py','--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--output',str(output)]
with log.open('x') as handle:
 child=subprocess.Popen(argv,cwd=root,start_new_session=True,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps(dict(pid=child.pid,host='2026',gpu=1,output=str(output),log=str(log),source_sha256={sources!r},started=time.time(),command=argv,status='TWO_REAL_PREFLIGHT_GATES_STARTED_THEN_ONE_FRESH50_MSVR',budget_epochs=50,planned_engineering_updates=3,official_test_uses=0)))
'''
receipt=json.loads(remote_python('2026',start));receipt['availability']=availability;receipt['observed_at']=datetime.now().isoformat(timespec='seconds')
target=PROJECT/'results/preflight/axis_collaboration_v6_projected_ampfix_launch.json';assert not target.exists()
target.write_bytes(json.dumps(receipt,indent=2).encode('utf-8'))
print('V6_PROJECTED26_AMPFIX_STARTED',json.dumps({key:receipt[key] for key in ('pid','host','gpu','status')}),flush=True)
