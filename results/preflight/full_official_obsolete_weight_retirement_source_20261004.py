import hashlib,json,shutil,time
from pathlib import Path
root=Path('/data/gaob/Re-ID/DeMo-DualAxis').resolve()
checked=[{'copied': 'runs/first_comparison/MSVR310_dual_s42/best.pth', 'canonical': 'runs/dynamic_amp_comparison/MSVR310_dual_s42/best.pth', 'copied_sha256': '66d8f5f09ec678ddb45333e0b4b055f3ab8174ba293d293edef9d906f660be73', 'canonical_sha256': 'f1404b12c35283a5288075987aa56925127299a703dca2009d32a234814cc26a', 'bytes': 418179683}, {'copied': 'runs/first_comparison/RGBNT100_demo_s42/best.pth', 'canonical': 'runs/dynamic_amp_comparison/RGBNT100_demo_s42/best.pth', 'copied_sha256': 'a318e72bf748ca09ece4976ebebd8efa7ab450763861b3652f4f6c3f519f1c61', 'canonical_sha256': 'a53c9aba1d7a675aa71ba380fd0b533aa63f9830366856bcbd68891f04d5a01e', 'bytes': 391737905}, {'copied': 'runs/first_comparison/MSVR310_ordinary_s42/best.pth', 'canonical': 'runs/dynamic_amp_comparison/MSVR310_ordinary_s42/best.pth', 'copied_sha256': 'd3fc48eb10270fa90863a0fd6e08b2c6eea8c53e83e82dc4ba9cb2896dd2b99b', 'canonical_sha256': 'd3fc48eb10270fa90863a0fd6e08b2c6eea8c53e83e82dc4ba9cb2896dd2b99b', 'bytes': 418179683}, {'copied': 'runs/first_comparison/RGBNT100_dual_s42/best.pth', 'canonical': 'runs/dynamic_amp_comparison/RGBNT100_dual_s42/best.pth', 'copied_sha256': '0cc89b12729ef70faf41b636bc66db1883fed7a342a09d287837e778aa6878bd', 'canonical_sha256': 'af19a0276420cc4e5a290d6e7c7cba512ac61098fa3672d5adad5c0d3b2970a2', 'bytes': 415025763}]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
processes=[p.joinpath('cmdline').read_bytes().split(b'\0') for p in Path('/proc').glob('[0-9]*') if p.joinpath('cmdline').exists()]
receipt=root/'results/preflight/full_official_obsolete_weight_retirement_20261004.json'
assert not receipt.exists()
rows=[]
for row in checked:
 copied=(root/row['copied']).resolve();canonical=(root/row['canonical']).resolve()
 assert copied.is_relative_to(root/'runs/first_comparison') and copied.name=='best.pth'
 assert canonical.is_relative_to(root/'runs/dynamic_amp_comparison') and canonical.name=='best.pth'
 assert not any(str(copied.parent).encode() in tokens or str(copied.parent.relative_to(root)).encode() in tokens for tokens in processes)
 assert copied.stat().st_size==row['bytes'] and digest(copied)==row['copied_sha256']
 assert digest(canonical)==row['canonical_sha256']
 completed=json.loads(canonical.parent.joinpath('result.json').read_text())
 assert completed['status']=='COMPLETE' and completed['epochs']==50 and json.loads(canonical.parent.joinpath('exit.json').read_text())['exit_code']==0
 if row['copied_sha256']==row['canonical_sha256']:
  reason='Exact duplicate of retained completed50 canonical checkpoint'
 else:
  assert not copied.parent.joinpath('result.json').exists()
  status=json.loads(copied.parent.joinpath('status.json').read_text());assert status['epoch']<50
  reason='Superseded nonterminal first attempt at epoch'+str(status['epoch'])+'; completed50 canonical rerun retained'
 rows.append(dict(row,reason=reason))
before=shutil.disk_usage(root).free
for row in rows:
 path=root/row['copied'];path.unlink();assert not path.exists()
 assert digest(root/row['canonical'])==row['canonical_sha256']
result=dict(status='RETIRED_FOUR_SUPERSEDED_OR_DUPLICATE_WEIGHTS',removed=rows,removed_bytes=sum(r['bytes'] for r in rows),free_before=before,free_after=shutil.disk_usage(root).free,at=time.time(),active_full_official_best_weights_untouched=True,logs_results_unchanged=True)
receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
