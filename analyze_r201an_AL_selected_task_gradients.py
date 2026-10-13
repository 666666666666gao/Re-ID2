"""Summarize actual AN training-only gradients; no inference or optimization."""
import csv
import json
import math
from pathlib import Path

P = Path(__file__).resolve().parent
TAG = 'r201an_AL_selected_task_gradients_20261013'
PAIRS = ('primary_CE_vs_hinge', 'fused_primary_vs_M_aux', 'fused_primary_vs_F_aux',
    'M_aux_vs_F_aux', 'fused_primary_vs_M_F_aux', 'fused_primary_vs_contribution', 'fused_primary_vs_full_total')


def main():
    root = P / 'results' / TAG
    controller = json.loads((root / 'controller_result.json').read_text(encoding='utf-8'))
    assert controller['status'] == 'COMPLETE' and controller['neural_forwards'] == 10
    summaries = []
    for job in controller['full']:
        path = root / 'full' / (job['name'] + '.json')
        result = json.loads(path.read_text(encoding='utf-8'))
        assert result['status'] == 'COMPLETE' and result['batches'] == 4
        for role in result['roles']:
            values = [r['deploy_parameter_roles'][role] for r in result['rows']]
            row = dict(variant=result['variant'], selected_epoch=result['selected_epoch'], role=role, batches=4)
            for loss in values[0]['gradient_l2']:
                row[loss + '_gradient_L2_mean'] = math.fsum(r['gradient_l2'][loss] for r in values) / 4
            for pair in PAIRS:
                cosines = [r[pair]['cosine'] for r in values if r[pair]['cosine'] is not None]
                row[pair + '_defined'] = len(cosines)
                row[pair + '_cosine_mean'] = math.fsum(cosines) / len(cosines) if cosines else None
                row[pair + '_negative'] = sum(v < 0 for v in cosines)
            summaries.append(row)
    output = root / 'analysis'
    output.mkdir(exist_ok=False)
    with (output / 'task_gradient_role_summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    proof = dict(status='ACTUAL_AN_AL_SELECTED_TRAIN_ONLY_GRADIENT_SUMMARY_COMPLETE',
        controls=2, full_batches=8, total_neural_forwards=10, optimizer_updates=0,
        checkpoint_writes=0, official_query_gallery_neural_uses=0, roles=summaries,
        new_neural_calls=0, limits='Four same-order training batches per fixed AL selected control; different selected epochs. Means and negative counts describe these local observations, not training-wide conflict, causal failure, M/F source incompatibility or new retrieval performance. None is an actually disconnected/zero gradient. No gradient arbitration is applied.')
    (output / 'result.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=proof['status'], controls=2, full_batches=8)))


if __name__ == '__main__':
    main()
