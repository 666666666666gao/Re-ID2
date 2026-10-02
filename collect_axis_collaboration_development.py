"""Read-only V3 development watcher and terminal text intake, never test scoring."""
import argparse
from datetime import datetime
import json
import time

from collect_results import PROJECT, HOSTS, copy_file, remote_python
from launch_axis_collaboration_development import SCHEDULE, run_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    base = PROJECT/'results/axis_collaboration_v3_development'
    base.mkdir(parents=True, exist_ok=True)
    while True:
        code = '''import json
from pathlib import Path
root=Path('/data2/gb/Re-ID/DeMo-DualAxis/runs/axis_collaboration_v3_development')
records={}
for run in root.glob('*_s42'):
    record={}
    for key in ('launch','run','status','best','result','exit'):
        path=run/(key+'.json')
        if path.exists():record[key]=json.loads(path.read_text())
    record['files']=[name for name in ('stdout.log','epochs.csv','batch_orders.jsonl') if (run/name).exists()]
    records[run.name]=record
print(json.dumps(records))
'''
        records = json.loads(remote_python('2025', code))
        rows = []
        for gpu, jobs in SCHEDULE.items():
            for dataset, variant, contribution in jobs:
                name = run_name(dataset, variant, contribution)
                record = records.get(name, {})
                exit_code = record.get('exit', {}).get('exit_code')
                phase = 'FAILED' if exit_code not in (None, 0) else ('COMPLETE' if exit_code == 0 and record.get('result', {}).get('status') == 'COMPLETE' else ('TRAINING' if 'launch' in record else 'WAIT_PREDECESSOR'))
                status = record.get('status', {})
                rows.append({'run': name, 'gpu': gpu, 'status': phase, 'epoch': status.get('epochs', status.get('epoch', 0)), 'best': record.get('best')})
                folder = base/name
                if exit_code is not None and not (folder/'intake.json').exists():
                    source = f'{HOSTS["2025"][0]}/runs/axis_collaboration_v3_development/{name}'
                    for key in ('launch','run','status','best','result','exit'):
                        if key in record:
                            copy_file('2025', f'{source}/{key}.json', folder/f'{key}.json')
                    for filename in record['files']:
                        copy_file('2025', f'{source}/{filename}', folder/filename)
                    (folder/'intake.json').write_text(json.dumps({'exit_code': exit_code}), encoding='utf-8')
        summary = {'time': datetime.now().astimezone().isoformat(timespec='seconds'), 'rows': rows,
                   'scope': 'Nine parameter-matched seed42 dev runs plus MSVR contribution-off dev comparison; no official-test efficacy claim.'}
        (base/'status.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print('AXIS_COLLABORATION_DEVELOPMENT', json.dumps(summary), flush=True)
        if all(row['status'] == 'COMPLETE' for row in rows) or any(row['status'] == 'FAILED' for row in rows) or not args.watch:
            return
        time.sleep(240)


if __name__ == '__main__':
    main()
