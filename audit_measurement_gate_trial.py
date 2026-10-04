"""M3a exact M2b sampling/parameters plus the installed-GT 3087-case audit."""
import json
from pathlib import Path
import sys

import audit_identity_alignment_trial as parent


if __name__ == '__main__':
    root = Path(sys.argv[sys.argv.index('--root') + 1])
    tensor = json.loads((root / 'preflight/tensor/result.json').read_text())
    assert tensor['status'] == 'PASS_MEASUREMENT_GATE_GRADIENT_CONTRACT'
    assert tensor['full_and_all_partial_task_gradient_blocked'] and tensor['regression_gradient_real_nonzero']
    for variant in ('axis_shared', 'frequency_shared', 'twins_shared'):
        run = root / 'original_mean/development' / ('MSVR310_' + variant + '_s42')
        trained = json.loads((run / 'result.json').read_text())
        assert trained['arguments']['gate_gradient_mode'] == 'measurement_only'
        orders = [json.loads(line) for line in (run / 'batch_orders.jsonl').read_text().splitlines()]
        assert all(row['gate_gradient_mode'] == 'measurement_only' for row in orders)
    parent.main()
    path = root / 'independent_cpu_audit.json'
    result = json.loads(path.read_text())
    result['M3a_gate_gradient_contract_verified'] = True
    result['reference_scope'] = 'Original M2b same sampling and parameter controls; only task gradient into predictor removed'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('MEASUREMENT_GATE_ALL3087_CPU_PASS', flush=True)
