import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, remote_python
HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')

def inspect(host):
    root, _ = HOSTS[host]
    code = f'''import json,shutil,subprocess
from pathlib import Path
root=Path({root!r})
lines=subprocess.check_output(['ps','-eo','pid=,ppid=,etime=,stat=,args='],text=True).splitlines()
selected=[line[:1400] for line in lines if 'python' in line and ('DeMo-DualAxis' in line or 'zlib' in line or 'root=Path' in line)]
print(json.dumps(dict(processes=selected,free_bytes=shutil.disk_usage(root).free,cleanup_receipt_exists=(root/'results/all_closed_nonbest_weight_cleanup_20261003.json').exists())))
'''
    return host, json.loads(remote_python(host, code))

with ThreadPoolExecutor(max_workers=3) as pool:
    records = dict(pool.map(inspect, ('2025', '2026', '2027')))
path = PROJECT/'results/preflight/nonbest_cleanup_runtime_state.json'
path.write_bytes(json.dumps(dict(observed_at=datetime.now().isoformat(timespec='seconds'), hosts=records), indent=2).encode())
for host, record in records.items():
    print(host, json.dumps(record), flush=True)
