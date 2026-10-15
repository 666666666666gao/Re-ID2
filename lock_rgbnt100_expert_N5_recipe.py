"""Lock the bounded five-seed candidate only after current BC results are closed/audited."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parent
PF=PROJECT/'results/preflight'
TAGS=dict(AV='r201av_RGB_frequency_value_decay_20261014',BC='r201bc_RGB_reference_interaction_20261015')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    output=PF/'rgbnt100_expert_N5_selected_recipe_lock_20261015.json';assert not output.exists()
    intake_path=PF/(TAGS['BC']+'_complete_intake.json')
    intake=load(intake_path)
    assert intake['status']=='ACTUAL_BC_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    assert intake['terminal']['all_children_and_controller_absent']
    audit_path=PF/'r201bc_closed_normal_integrity_audit_20261015.json';audit=load(audit_path)
    assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
    folders={key:PROJECT/'results'/tag/'RGBNT100/training'/('RGBNT100_r201'+key.lower()+'_axis_shared_s42') for key,tag in TAGS.items()}
    values={key:load(folder/'result.json') for key,folder in folders.items()}
    assert all(d['status']=='COMPLETE' and d['epochs']==50 and d['arguments']['seed']==42 for d in values.values())
    av,bc=(values[key]['full_metrics'] for key in ('AV','BC'))
    selected='BC' if bc['mAP']>av['mAP'] and bc['Rank-1']>=av['Rank-1'] else 'AV'
    original=load(PROJECT/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42/result.json')['full_metrics']
    targets={key:original[key]+1 for key in ('mAP','Rank-1')}
    gaps={key:targets[key]-values[selected]['full_metrics'][key] for key in targets}
    assert max(max(0,value) for value in gaps.values())<=1., 'Candidate is not within the predeclared near-target range'
    assert any(value>0 for value in gaps.values()), 'Normal double+1 is already met; no bounded tuning is needed'
    family='rgbnt100_expert_N5_'+selected+'_20261015'
    seed42={variant:str((PROJECT/'results'/TAGS[selected]/'RGBNT100/training'/('RGBNT100_r201'+selected.lower()+'_'+variant+'_s42')).relative_to(PROJECT)).replace('\\','/') for variant in ('frequency_shared','axis_shared')}
    record=dict(status='ACTUAL_CLOSED_RECIPE_LOCKED_FOR_BOUNDED_EXPERT_N5',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        selected_recipe=selected,expert_seeds=[42,43,44,45,46],teacher_seed=42,seed42_reused=True,
        recipe_choice_rule='BC only if its completed axis mAP is strictly above retained AV and its Rank-1 is at least AV; otherwise retain AV. No architecture or parameter retuning between seeds.',
        main_seed_selection_rule='Among the five mAP-best checkpoints: prioritize satisfying both original-DeMo+1 thresholds, then larger mAP, then smaller seed. Apply the same tuple to ordinary and axis; retain all five rows and also disclose the unfiltered maximum-mAP run.',
        parent_axis_six={key:d['full_metrics'] for key,d in values.items()},original_DeMo_six=original,targets=targets,remaining_gaps=gaps,
        parameters=values[selected]['parameters'],trainable_parameters=values[selected]['trainable_parameters'],descriptor_dim=5120,
        local_family_root='results/'+family,remote_family_root='/data/gaob/Re-ID/DeMo-DualAxis/runs/'+family,
        seed42_local_runs=seed42,BC_intake_path=str(intake_path.relative_to(PROJECT)),BC_intake_sha256=hashlib.sha256(intake_path.read_bytes()).hexdigest(),
        BC_audit_path=str(audit_path.relative_to(PROJECT)),BC_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        new_formal_models_max=8,normal_only=True,missing_evaluated=False,new_neural_calls=0,new_optimizer_updates=0,
        limits='Post-BC method lock, before any new seed43..46 result. Reuses one benchmark-selected original teacher42 and completed recipe42 pair; conditional repeats, not five end-to-end pipelines or an untouched final test. Selection and every actual attempt are disclosed.')
    output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],selected_recipe=selected,remaining_gaps=gaps)),flush=True)


if __name__=='__main__':main()
