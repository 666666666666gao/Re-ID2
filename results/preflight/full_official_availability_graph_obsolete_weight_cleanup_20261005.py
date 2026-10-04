"""Retire five completed obsolete negative checkpoints; preserve current controls."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

root, _ = HOSTS['2026']
pf = PROJECT / 'results/preflight'
name = 'full_official_availability_graph_obsolete_weight_cleanup_20261005.json'
receipt = pf / name
assert not receipt.exists()
intake = json.loads((pf / 'full_official_availability_graph_obsolete_negative_weights_intake_20261005.json').read_text(encoding='utf-8'))
assert len(intake) == 5
result = json.loads(remote_python('2026', f'''import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
root=Path({root!r}).resolve();runs=(root/'runs').resolve()
targets={list(intake)!r}
expected={dict((p, r['result.json']['best']) for p, r in intake.items())!r}
protected=[]
for campaign,count in [('full_official_baselines_20261004',6),('full_official_control_m3b_20261004',6),('full_official_modality_outlet_m4_20261005',3)]:
 files=sorted((runs/campaign/'training').glob('*/best.pth'))
 assert len(files)==count,(campaign,len(files))
 protected.extend(files)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
before={{str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in protected}}
manifest=[]
for relative in targets:
 path=(root/relative).resolve()
 assert path.is_relative_to(runs) and path.name=='best.pth' and path not in protected
 assert path.stat().st_size==399087701
 current=json.loads((path.parent/'result.json').read_text())
 assert current['best']==expected[relative]
 assert current['status']=='COMPLETE' and current['epochs']==50
 assert current['fit_records']==672 and current['dev_records']==360 and current['dev_queries']==210
 manifest.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest(path),best=current['best'],retired_protocol='identity-heldout fit672/dev360; completed negative and superseded by full-official protocol'))
usage=os.statvfs(root);free_before=usage.f_bavail*usage.f_frsize
receipt=root/'results/preflight'/{name!r}
assert not receipt.exists()
row=dict(status='VERIFIED_FIVE_OBSOLETE_NEGATIVE_CHECKPOINTS_PENDING_UNLINK',observed_at=datetime.now(timezone.utc).isoformat(),targets=manifest,protected_best_checkpoints=before,free_bytes_before=free_before)
receipt.write_text(json.dumps(row,indent=2)+'\\n')
for relative in targets:(root/relative).unlink()
assert all(not (root/relative).exists() for relative in targets)
after={{str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in protected}}
assert after==before
usage=os.statvfs(root)
row.update(status='FIVE_OBSOLETE_NEGATIVE_CHECKPOINTS_REMOVED_FIFTEEN_CURRENT_BEST_REFERENCES_UNCHANGED',removed_bytes=sum(x['bytes'] for x in manifest),removed_files=len(manifest),free_bytes_after=usage.f_bavail*usage.f_frsize,protected_after=after,text_results_preserved=True,raw_distances_not_touched=True)
receipt.write_text(json.dumps(row,indent=2)+'\\n');print(json.dumps(row))
'''))
receipt.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
assert result['removed_files'] == 5 and result['removed_bytes'] == 1995438505
assert result['protected_after'] == result['protected_best_checkpoints']
assert remote_python('2026', 'from pathlib import Path;import hashlib;print(hashlib.sha256(Path(' + repr(root + '/results/preflight/' + name) + ').read_bytes()).hexdigest())').strip() == hashlib.sha256(receipt.read_bytes()).hexdigest()
print(result['status'], result['removed_bytes'], result['free_bytes_after'], flush=True)
