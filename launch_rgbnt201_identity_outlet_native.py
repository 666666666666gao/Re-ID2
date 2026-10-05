"""Native gate only; each physical GPU runs its two paired outlet modes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

from gpu_thermal_execute import execute
from run_experiment import write_json


def main():
    parser=argparse.ArgumentParser()
    for key in ('data-root','pretrained','anchor-run-dir','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),
        arguments=vars(args),gpus=[2,3],formal50_started=0,temperature_power_control=False))
    def lane(variant,gpu):
        return execute([sys.executable,'-u','verify_rgbnt201_identity_outlet.py',
            '--data-root',args.data_root,'--pretrained',args.pretrained,
            '--anchor-run-dir',args.anchor_run_dir,'--variant',variant,
            '--output',str(root/variant)],root,variant,gpu)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,variant,gpu) for variant,gpu in (('frequency_shared',2),('axis_shared',3))]
        exits=[future.result() for future in futures]
    rows={variant:json.loads((root/variant/'result.json').read_text()) for variant in ('frequency_shared','axis_shared')}
    assert all(row['status']=='PASS_RGBNT201_IDENTITY_OUTLET_NATIVE3_PER_MODE_FULL_READOUT' for row in rows.values())
    assert rows['frequency_shared']['controls']['0']['batch_names']==rows['axis_shared']['controls']['0']['batch_names']
    assert len({(row['controls']['0']['parameters'],row['controls']['0']['trainable_parameters']) for row in rows.values()})==1
    assert all(row['descriptor_dim']==5120 and row['actual_updates']==6 and row['amp_skips']==0 for row in rows.values())
    assert not list(root.rglob('*.pth'))
    write_json(root/'controller_result.json',dict(status='COMPLETE_IDENTITY_OUTLET_NATIVE_ONLY',
        exits=exits,controls=rows,actual_updates=12,formal50_started=0,
        query_count=836,gallery_count=836,training_heldout_identities=0,new_weight_files=0,finished=time.time()))


if __name__=='__main__':
    main()
