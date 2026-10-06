"""Normal selected-checkpoint query interactions; not calibrated or causal contributions."""
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import random

from analyze_full_official_baseline_pairs import query_rows
from analyze_identity_coordinate_three_normal import table

PROJECT = Path(__file__).resolve().parent
STATES = ('00','10','01','11')
KEYS = ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')


def interval(values):
    ordered = sorted(values)
    return [ordered[int(.025*(len(ordered)-1))],ordered[int(.975*(len(ordered)-1))]]


def main():
    root = PROJECT/'results/r201g_normal_states_20261006'
    actual = json.loads((PROJECT/'results/preflight/r201g_normal_states_actual_20261006.json').read_text())
    assert actual['state_cases']==8 and actual['optimizer_updates']==actual['checkpoint_writes']==0
    closed = json.loads((root/'normal_state_analysis/result.json').read_text())
    assert closed['status']=='ACTUAL_R201G_SELECTED_NORMAL_FOURSTATE_CPU_READOUT_COMPLETE'
    output = root/'joint_query_utility_analysis'
    assert not output.exists()
    summaries, details, outcomes = [], [], []
    for variant in ('frequency_shared','axis_shared'):
        folder = root/('RGBNT201_r201g_'+variant+'_s42')
        reads = {state:query_rows(folder/('state_'+state+'.csv'),836) for state in STATES}
        assert all(all(all(row[key]==base[key] for key in KEYS) for row,base in zip(rows,reads['00'])) for rows in reads.values())
        counts = Counter()
        identities = defaultdict(list)
        for index, base in enumerate(reads['00']):
            aps = {state:float(reads[state][index]['AP']) for state in STATES}
            correct = {state:int(reads[state][index]['first_match'])==1 for state in STATES}
            mask = ''.join(str(int(correct[state])) for state in STATES)
            counts[mask] += 1
            row = dict(variant=variant,**{key:base[key] for key in ('query_index','name','identity','camera','scene')},
                outcomes_B_M_F_J=mask,base_AP=aps['00'],modality_state_AP=aps['10'],frequency_state_AP=aps['01'],joint_AP=aps['11'],
                delta_M_given_F_AP_pp=100*(aps['11']-aps['01']),delta_F_given_M_AP_pp=100*(aps['11']-aps['10']),
                empirical_AP_interaction_pp=100*(aps['11']-aps['10']-aps['01']+aps['00']),
                delta_joint_base_AP_pp=100*(aps['11']-aps['00']),delta_joint_base_Rank1_pp=100*(int(correct['11'])-int(correct['00'])),
                empirical_Rank1_interaction_pp=100*(int(correct['11'])-int(correct['10'])-int(correct['01'])+int(correct['00'])),
                only_joint_correct=correct['11'] and not correct['10'] and not correct['01'],
                base_wrong_only_joint_correct=not correct['00'] and correct['11'] and not correct['10'] and not correct['01'],
                any_single_correct_joint_wrong=not correct['11'] and (correct['10'] or correct['01']),
                M_correct_F_wrong=correct['10'] and not correct['01'],F_correct_M_wrong=correct['01'] and not correct['10'])
            details.append(row)
            identities[int(base['identity'])].append(row)
        chosen = [row for row in details if row['variant']==variant]
        keys = ('delta_joint_base_AP_pp','delta_joint_base_Rank1_pp','delta_M_given_F_AP_pp','delta_F_given_M_AP_pp','empirical_AP_interaction_pp','empirical_Rank1_interaction_pp')
        blocks = {identity:dict(count=len(rows),sums={key:math.fsum(row[key] for row in rows) for key in keys}) for identity,rows in identities.items()}
        rng = random.Random(20261006)
        bootstrap = {key:[] for key in keys}
        ids = sorted(blocks)
        for _ in range(2500):
            sample = [blocks[rng.choice(ids)] for _ in ids]
            total = sum(row['count'] for row in sample)
            for key in keys:
                bootstrap[key].append(math.fsum(row['sums'][key] for row in sample)/total)
        summary = dict(variant=variant,queries=836,identities=len(ids),
            only_joint_correct=sum(row['only_joint_correct'] for row in chosen),
            base_wrong_only_joint_correct=sum(row['base_wrong_only_joint_correct'] for row in chosen),
            any_single_correct_joint_wrong=sum(row['any_single_correct_joint_wrong'] for row in chosen),
            M_correct_F_wrong=sum(row['M_correct_F_wrong'] for row in chosen),F_correct_M_wrong=sum(row['F_correct_M_wrong'] for row in chosen),
            positive_AP_interaction_queries=sum(row['empirical_AP_interaction_pp']>0 for row in chosen),
            negative_AP_interaction_queries=sum(row['empirical_AP_interaction_pp']<0 for row in chosen),
            mean_pp={key:math.fsum(row[key] for row in chosen)/836 for key in keys},
            identity_cluster_bootstrap_95_intervals_pp={key:interval(bootstrap[key]) for key in keys})
        original = next(row for row in closed['comparisons'] if row['improved_variant']==row['reference_variant']==variant and row['improved_state']=='11' and row['reference_state']=='00')
        assert abs(summary['mean_pp']['delta_joint_base_AP_pp']-original['mAP'])<1e-8
        assert abs(summary['mean_pp']['delta_joint_base_Rank1_pp']-original['Rank-1'])<1e-8
        summaries.append(summary)
        outcomes.extend(dict(variant=variant,outcomes_B_M_F_J=mask,queries=count) for mask,count in sorted(counts.items()))
    assert len(details)==1672 and sum(row['queries'] for row in outcomes)==1672
    output.mkdir()
    table(output/'query_contributions.csv',details)
    table(output/'four_state_rank1_outcomes.csv',outcomes)
    result = dict(status='ACTUAL_G_201_SELECTED_NORMAL_QUERY_INTERACTION_CPU_DIAGNOSTIC',completed_at=datetime.now().isoformat(timespec='seconds'),
        variants=summaries,checked_repeated_query_rows=1672,unique_query_records=836,bootstrap_draws=2500,bootstrap_seed=20261006,
        new_neural_calls=0,new_optimizer_updates=0,
        limits='Same selected seed42 checkpoints and officialbenchmark selection. Identity-cluster resampling of this selected query set is not paired training-seed confirmation or untouched-test evidence. M/F states contain the base plus one expert; not standalone experts or retrained single branches. Each state uses its corresponding gallery descriptor; references share GT record membership, not frozen feature coordinates. AP/Rank1 interaction is empirical, not information-theoretic or causal, not the calibrator training margin or a verified contribution prediction. Positive joint utility does not isolate psi, PI, conditioning or capacity.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
