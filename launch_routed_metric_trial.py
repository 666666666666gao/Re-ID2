"""One P1 routed-supervision trial: tensor contract, three native AMP updates, then fresh50."""
import argparse
import json
from pathlib import Path
import sys

from launch_axis_scaled import execute
from run_experiment import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--pretrained',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(exist_ok=False)
    preflight=output/'preflight';preflight.mkdir()
    argv=[sys.executable,'-u','verify_routed_metric_supervision.py','--data-root',args.data_root,'--pretrained',args.pretrained,
        '--output',str(preflight/'tensor')]
    rows=[execute(argv,preflight,'tensor',1)]
    assert json.loads((preflight/'tensor/result.json').read_text())['status']=='PASS_ROUTED_METRIC_SUPERVISION_TENSOR_CONTRACT'
    common=[sys.executable,'-u','run_routed_metric_experiment.py','--dataset','MSVR310','--variant','axis_metric_routed_fullref',
        '--seed','42','--contribution-weight','.05','--data-root',args.data_root,'--pretrained',args.pretrained]
    rows.append(execute(common+['--mode','smoke','--output',str(preflight/'smoke')],preflight,'smoke',1))
    assert json.loads((preflight/'smoke/smoke.json').read_text())['status']=='SMOKE_PASS'
    write_json(output/'preflight_result.json',dict(status='PASS',checks=rows))
    development=output/'development';development.mkdir()
    name='MSVR310_axis_metric_routed_fullref_s42'
    rows.append(execute(common+['--mode','train','--output',str(development/name)],development,name,1))
    result=json.loads((development/name/'result.json').read_text())
    assert result['status']=='COMPLETE' and result['epochs']==50
    assert result['retrieval_interface']=='metric_weighted_frequency_block'
    write_json(output/'controller_result.json',dict(status='COMPLETE',checks=rows,development_run=name))


if __name__=='__main__':main()
