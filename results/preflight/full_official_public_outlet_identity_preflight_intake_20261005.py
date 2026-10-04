"""Read closed native contracts and smokes after the sole observer reports PASS."""
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, remote_python

pf = PROJECT / 'results/preflight'
launch = json.loads((pf / 'full_official_public_outlet_identity_launch_20261005.json').read_text(encoding='utf-8'))['launch']
source = launch['output']
out = pf / 'full_official_public_outlet_identity_actual_preflight_20261005.json'
assert not out.exists()
observations = [json.loads(line) for line in (pf / 'full_official_public_outlet_identity_primary_observer_20261005.jsonl').read_text(encoding='utf-8').splitlines()]
assert observations[-1]['preflight_pass'] and not observations[-1]['failures']
assert all(row['contract_pass'] and row['smoke_pass'] for row in observations[-1]['runs'])
result = json.loads(remote_python('2026', f'''import json
from pathlib import Path
root=Path({source!r});preflight=json.loads((root/'preflight_result.json').read_text())
assert preflight['status']=='PASS' and len(preflight['runs'])==4
assert preflight['contract_actual_optimizer_updates']==16 and preflight['smoke_actual_optimizer_updates']==12
contracts,smokes={{}},{{}}
for row in preflight['runs']:
 name='MSVR310_public_identity_'+row['variant']+'_s42'
 contract=json.loads((root/'contract'/name/'result.json').read_text())
 smoke=json.loads((root/'preflight'/name/'smoke.json').read_text())
 assert contract['status']=='PASS_FULL_OFFICIAL_PUBLIC_IDENTITY_PARENT_AMP_STEP'
 assert contract['parent_step_loss_gradient_equivalence'] and contract['optimizer_update_consistency']
 assert len(contract['own_adam_update_checks'])==4 and all(r['actual_optimizer_updates']==1 for r in contract['own_adam_update_checks'])
 positive=contract['positive_public_identity_branch']
 assert positive['actual_public_outlet_inputs_exact'] and positive['classifier_inputs_and_official_training_labels_exact']
 assert positive['existing_head_calls']==3 and positive['intended_extra_shared_BN_updates']==2
 assert positive['backbone_calls']==4 and positive['projection_calls']==2 and positive['added_backbone_calls']==0
 assert positive['other_buffers_exact'] and positive['RNG_exact']
 assert smoke['status']=='SMOKE_PASS' and smoke['steps']==3 and smoke['strict_reload_equal']
 assert smoke['training_heldout_identities']==0 and all(smoke['gradients'].values())
 assert sum(r['optimizer_updated'] for r in smoke['details'])==3
 assert smoke['attempts']-3==smoke['amp_skipped_steps']
 for phase in ('contract','preflight'):
  assert json.loads((root/phase/(name+'_exit.json')).read_text())['exit_code']==0
 contracts[row['variant']],smokes[row['variant']]=contract,smoke
print(json.dumps(dict(preflight=preflight,contracts=contracts,smokes=smokes)))
'''))
assert set(result['contracts']) == set(result['smokes']) == {'demo_shared','axis_shared','frequency_shared','twins_shared'}
result.update(status='FOUR_REAL_NATIVE_PUBLIC_OUTLET_CONTRACTS_AND_TWELVE_TRUE_SMOKE_UPDATES_PASSED',
    collected_at=datetime.now().isoformat(timespec='seconds'), source=source,
    actual_contract_optimizer_updates=sum(sum(r['actual_optimizer_updates'] for r in c['own_adam_update_checks']) for c in result['contracts'].values()),
    actual_smoke_optimizer_updates=sum(s['steps'] for s in result['smokes'].values()),
    actual_smoke_amp_skips=sum(s['amp_skipped_steps'] for s in result['smokes'].values()),
    training_started_snapshot=observations[-1], actual_completed_fresh50=0, new_neural_execution=0,
    limits='Reads completed native proof and the sole observer snapshot only; does not rerun an update, smoke or training, and is not final benchmark performance.')
assert result['actual_contract_optimizer_updates'] == 16 and result['actual_smoke_optimizer_updates'] == 12
out.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
print('FULL_OFFICIAL_PUBLIC_IDENTITY_ACTUAL_PREFLIGHT_COLLECTED', json.dumps(dict(
    contract_updates=16, smoke_updates=12, smoke_amp_skips=result['actual_smoke_amp_skips'])), flush=True)
