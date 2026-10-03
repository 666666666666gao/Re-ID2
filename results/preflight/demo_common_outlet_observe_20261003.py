import json
from pathlib import Path
import sys
import time

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, remote_python

receipt=json.loads((PROJECT/'results/preflight/common_outlet_scale_2026_launch.json').read_text())
started=time.time()
while time.time()-started<1200:
    time.sleep(240)
    code=f'''import json
from pathlib import Path
root=Path({receipt['output']!r});process=Path('/proc/{receipt['pid']}')
live=process.exists() and (process/'stat').read_text().split()[2]!='Z'
result=json.loads((root/'controller_result.json').read_text()) if (root/'controller_result.json').exists() else None
rows={{str(p.relative_to(root)):json.loads(p.read_text()) for p in root.rglob('*_exit.json')}}
print(json.dumps(dict(pid={receipt['pid']},live=live,result=result,exits=rows,log_tail=Path({receipt['log']!r}).read_text()[-2500:])))
'''
    snapshot=json.loads(remote_python('2026',code))
    (PROJECT/'results/preflight/common_outlet_scale_latest_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    print('OUTLET_OBSERVATION',json.dumps(snapshot),flush=True)
    if not snapshot['live']:
        terminal=PROJECT/'results/preflight/common_outlet_scale_observer_terminal.json'
        assert not terminal.exists()
        terminal.write_text(json.dumps(snapshot,indent=2)+'\n')
        break
