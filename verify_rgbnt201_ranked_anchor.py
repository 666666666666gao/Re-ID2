"""Reuse unchanged native gates; exercise real ranked updates and Adam replay."""
import verify_rgbnt201_original_anchor as verifier
from run_experiment import write_json
from run_rgbnt201_ranked_anchor import step


def record(path, value):
    if value.get('status') == 'PASS_RGBNT201_ORIGINAL_ANCHOR_TWO_NATIVE_UPDATES':
        for row in value['controls'].values():
            assert row['detail']['rank_full_valid_queries'] > 0
            assert row['detail']['rank_partial_valid_queries'] > 0
        value['status'] = 'PASS_RGBNT201_RANKED_ANCHOR_TWO_NATIVE_UPDATES'
        value['ranking_objective_exercised'] = True
    write_json(path, value)


if __name__ == '__main__':
    verifier.step = step
    verifier.write_json = record
    verifier.main()
