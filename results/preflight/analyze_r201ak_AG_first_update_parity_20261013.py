"""Recount saved native first-step contracts; no runtime query or neural execution."""
from pathlib import Path
from datetime import datetime
import hashlib,json

PROJECT=Path(__file__).resolve().parents[2]
PF=PROJECT/'results/preflight'
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    proof=PF/'r201ak_native_and_formal_launch_actual_20261013.json'
    current=load(proof)
    assert current['status']=='ACTUAL_AK_NATIVE6_PASS_TWO_FORMAL_GPU23_WORKERS_LIVE'
    inputs={proof.relative_to(PROJECT).as_posix():hashlib.sha256(proof.read_bytes()).hexdigest()}
    rows=[]
    for variant in ('frequency_shared','axis_shared'):
        path=PROJECT/'results/r201ag_RGB_coordinate_metric_20261012/RGBNT100/native'/('RGBNT100_r201ag_'+variant+'_s42_smoke')/'smoke.json'
        original=load(path)
        actual=next(row['smoke'] for row in current['native']
            if row['launch']['command'][row['launch']['command'].index('--variant')+1]==variant)
        old,new=original['details'][0],actual['details'][0]
        assert original['status']==actual['status']=='SMOKE_PASS'
        assert old['freeze_identity_encoder'] is True and new['freeze_identity_encoder'] is False
        keys=sorted(set(old)-{'freeze_identity_encoder'})
        assert all(old[key]==new[key] for key in keys)
        assert new['adaptation_phase']=='outlets' and new['last_visual_block_effective_LR_factor']==0
        assert not any(new['last_visual_block_native_gradients'].values())
        assert new['original_core_eval_modes_fixed'] and new['original_core_buffers_retained']
        assert actual['registered_trainable_parameters']==8975238
        inputs[path.relative_to(PROJECT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(dict(variant=variant,contract_keys_checked=keys,
            exact_equal_logged_fields=len(keys),full_loss=old['full_loss'],
            names=old['names'],partial_set=old['partial_set'],
            recorded_whole_encoder_flag_changed=True,first_actual_phase='outlets',
            last_visual_block_gradients_absent=True,visual_LR_factor=0,
            interpretation='All shared first-native-step logged fields except the truthful whole-encoder flag equal AG exactly. New visual weights are registered but inactive.'))
    output=PF/'r201ak_AG_first_update_parity_actual_20261013.json'
    assert not output.exists()
    result=dict(status='ACTUAL_SAVED_NATIVE_FIRST_UPDATE_AG_AK_PARITY_PASS',
        at=datetime.now().astimezone().isoformat(timespec='seconds'),controls=2,rows=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256=inputs,
        new_neural_calls=0,new_optimizer_updates=0,new_remote_queries=0,
        limits='Saved log equality for one native update/control, not tensor/weight parity, five completed warmup epochs, full training equivalence, a gain, or a causal ablation. Subsequent AK native updates open the visual block and are intentionally different.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],controls=2,counts=[r['exact_equal_logged_fields'] for r in rows],full_loss=[r['full_loss'] for r in rows]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
