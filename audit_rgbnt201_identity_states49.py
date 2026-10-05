"""Installed-GT CPU recount for all4 states in every official condition."""
import argparse
import json
from pathlib import Path

import numpy as np

import audit_full_official49 as canonical


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--evaluation',required=True)
    args=parser.parse_args()
    # This independently checks full training, epoch selection and deployed11.
    canonical.main()
    run,out=Path(args.run_dir),Path(args.evaluation)
    trained=json.loads((run/'result.json').read_text())
    reported=json.loads((out/'result.json').read_text())
    root=Path(trained['arguments']['data_root'])
    query=canonical.installed_rows(root,'RGBNT201','query')
    gallery=canonical.installed_rows(root,'RGBNT201','gallery')
    assert len(query)==len(gallery)==836 and reported['state_cases']==196
    errors=[]
    for condition,states in reported['state_measurements'].items():
        assert set(states)=={'00','10','01','11'}
        with np.load(out/(condition+'.npz')) as raw:
            for state,summary in states.items():
                rows=canonical.recount(raw['distances' if state=='11' else 'distances_'+state],query,gallery,'RGBNT201')
                stem=condition if state=='11' else condition+'_state'+state
                _,error=canonical.verify(rows,summary,out/(stem+'.csv'),836)
                errors.append(error)
    assert len(errors)==196
    (out/'independent_fourstate_cpu_audit.json').write_text(json.dumps(dict(status='PASS',
        conditions=49,state_cases=196,query_records=836,gallery_records=836,
        max_sixmetric_error_pp=max(errors),optimizer_updates=0,
        limits='Installed GT filename recount, not semantic/new-method acceptance; repeated query sets.'),indent=2)+'\n')
    print('RGBNT201_FULL49_FOURSTATE_GT_CPU_PASS',flush=True)


if __name__=='__main__':
    main()
