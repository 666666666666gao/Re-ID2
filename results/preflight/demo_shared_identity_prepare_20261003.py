import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, remote_python

source = (PROJECT / 'missing_anytoany_development.py').read_text(encoding='utf-8')
source = source.replace("assert arguments.variant in ('demo','axis_mass_fullref')\n    if arguments.variant == 'demo':\n        from run_experiment import build\n    else:\n        from run_mass_experiment import build", "assert arguments.variant in ('axis_shared','frequency_shared','twins_shared','demo_shared')\n    from run_shared_identity_experiment import build")
source = source.replace("exit_file = run / 'exit.json' if arguments.variant == 'demo' else run.parent / (run.name + '_exit.json')", "exit_file = run.parent / (run.name + '_exit.json')")
source = source.replace("Single-seed frozen robustness diagnostic, not full official-train reproduction, independent training with missing inputs, or multi-seed acceptance.", "Single-seed robustness after matched partial-query/full-gallery training; not full official-train reproduction or multi-seed acceptance.")
target = PROJECT / 'missing_shared_identity_development.py'
assert not target.exists()
target.write_text(source, encoding='utf-8', newline='\n')
files = ('shared_identity_axis.py', 'run_shared_identity_experiment.py', 'verify_shared_identity_axis.py',
         'missing_shared_identity_development.py', 'launch_shared_identity_trial.py')
for name in files:
    ast.parse((PROJECT / name).read_text(encoding='utf-8'), filename=name)
hardware = json.loads(remote_python('2026', '''import json,shutil,subprocess
from pathlib import Path
root=Path('/data/gaob/Re-ID/DeMo-DualAxis')
rows=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
live=[row for row in rows if '/conda-envs/tri_reid/bin/python' in row and ' -c ' not in row]
print(json.dumps(dict(gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),live_neural_commands=live,disk_free_bytes=shutil.disk_usage(root).free,observed_at=__import__('time').time())))
'''))
plan = {
 'status': 'PREPARED_SOURCE_REVIEW_REQUIRED_NO_NEURAL_RUN', 'created': datetime.now().isoformat(timespec='seconds'),
 'objective': 'Three datasets mAP and Rank-1 each at least +2pp versus same-protocol DeMo, missing-modality coverage; current wave is mechanism development, not completion.',
 'evidence': ['V10 normal MSVR310 mAP45.991582/R159.047619 below DeMo47.607792/59.047619; do not repeat max-four-state hinge recipe.',
              'Frozen source masking: normal metrics unchanged exactly; V5 RGB100 R-query/full-gallery +30.737526mAP/+48.064R1, but some conditions regress.',
              '12/49 source-disjoint DeMo availability pairs have disjoint surviving private coordinates; V5 shared F tail is not strictly orthogonal.'],
 'recipe': {'shared_identity': 'mean raw CLS512+pooled patch512 from only available modalities in shared CLIP space, shared LN1024/Linear1024x512/GELU/LN512 projection',
            'private_base': 'original complete DeMo when all modalities available; absent backbone/reductions skipped; illegal S subset A relations and gates zero; absent relation BN experts skipped during partial forward',
            'metric': '5632D with separately normalized private5120 weight .75 and shared512 weight .25; frequency residual corrects nonzero shared coordinates; M/I correct legal private relation coordinates',
            'training': 'full supervised forward+backward then partial forward+backward, one optimizer step; same augmented images; six proper nonempty sets sampled uniformly per batch from independent seeded generator; same full batch detached gallery; GT positive excludes self, GT negative distinct IDs',
            'loss': 'all full-view original global/base losses retained; final fused ReID weight1; full M/F auxiliary .1 each; full shared ReID .25; partial label-smoothed CE .25 plus normalized cross-gallery soft triplet .5; existing contribution .05 for all augmented models; no V10 hinge',
            'controls': 'ordinary frequency and ordinary twins have exact same state keys, initial tensors, total/trainable parameters and5632D as dual axis; DeMo gets same shared interface, partial augmentation and losses except expert-specific terms',
            'fixed_training': 'MSVR310 fit/dev identity split, public CLIP, seed42, B64, AdamW existing cfg, native AMP512, fresh50, earliest dev-mAP tie wins; no official test'},
 'jobs': [dict(dataset='MSVR310', variant=v, gpu=i, host='2026') for i,v in enumerate(('axis_shared','frequency_shared','twins_shared','demo_shared'))],
 'gates': ['fresh same-family Astra/max source-only review before deployment', 'actual CUDA contract: exact augmented initial states/params, all7 availability banks finite and invalid coordinates zero, shared .25 metric, invalid expert BN unchanged, stopped gallery and GT reference legality', 'all4 smoke3 real updates and finite nonzero gradients, strict in-memory checkpoint reload before any fresh50', 'four50 epoch real update/AMP skip logs, full six metrics and CMC1..50/perquery/subgroups; strict best reload; all4 frozen49 conditions using same GT and earliest dev-best'],
 'acceptance': 'Original baseline +2mAP/+2R1 across all3 and missing not yet established; ordinary frequency/twins matched comparison required; this single dataset/seed wave cannot establish final success.',
 'global_max_parallel_runs':4, 'poll_seconds':240, 'hardware':hardware,
 'retention': 'Only best.pth per training run; no initial/last/smoke weights; raw logs and negative results retained; publicCLIP and earlier baseline/controls untouched.',
 'sources': {name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files}
}
path=PROJECT/'results/preflight/shared_identity_plan.json'
assert not path.exists()
path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(plan=str(path),sources=plan['sources'],hardware=hardware)),flush=True)
