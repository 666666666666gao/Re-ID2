"""Collect private MSVR310 diagnostic sheets and original JPEGs, without NN."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex

from collect_results import OPTIONS
from collect_rgbnt100_saved_error_images_CPU import command

PROJECT = Path(__file__).resolve().parent
ANALYSIS = PROJECT / 'results/msvr_saved_error_cases_CPU_20261015'
LOCAL = Path('C:/Users/gb/.codex_tmp/MSVR310_saved_error_images_CPU_20261015')
REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis/diagnostics/msvr_saved_error_images_CPU_20261015'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'


def main():
    output = PROJECT / 'results/preflight/msvr_saved_error_images_CPU_collection_actual_20261015.json'
    assert not output.exists() and not LOCAL.exists()
    analysis = json.loads((ANALYSIS / 'result.json').read_text('utf-8'))
    assert analysis['status'] == 'ACTUAL_SAVED_MSVR310_FIVE_OUTLET_NEAREST_MATCH_CPU_ANALYSIS_NO_NN'
    selection_path = ANALYSIS / 'selected_visual_cases.json'
    cases = json.loads(selection_path.read_text('utf-8'))['cases']
    code = f'''import hashlib,json,os,sys,zipfile
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and 'torch' not in sys.modules
root=Path('/data/gaob/Re-ID/dataset/MSVR310')
output=Path({REMOTE!r});assert not output.exists();output.mkdir(parents=True)
cases={cases!r}
maps={{s:{{int(p.name):p.name for p in (root/s).iterdir() if p.is_dir()}} for s in ('query3','bounding_box_test')}}
font=ImageFont.load_default();files={{}};sources=[];originals={{}}
for start in range(0,len(cases),6):
 rows=cases[start:start+6]
 sheet=Image.new('RGB',(2304,48+166*len(rows)),(20,24,30));draw=ImageDraw.Draw(sheet)
 for column,title in enumerate(('QUERY   RGB | NIR | TIR','WRONG MATCH   RGB | NIR | TIR','CORRECT ID   RGB | NIR | TIR')):
  draw.text((column*768+8,12),title,fill='white',font=font)
 for row_index,case in enumerate(rows):
  y=48+row_index*166
  for column,role in enumerate(('query','wrong','correct')):
   name=case[role+'_name'];identity=case[role+'_identity']
   assert Path(name).name==name and int(name[:4])==identity
   split='query3' if role=='query' else 'bounding_box_test'
   color='white' if role=='query' else '#ff8c8c' if role=='wrong' else '#8cffb0'
   label='case'+str(case['case'])+' '+case['comparison']+' '+case['cohort']+' '+name
   draw.text((column*768+8,y+4),label,fill=color,font=font)
   for mi,modality in enumerate(('vis','ni','th')):
    path=root/split/maps[split][identity]/modality/name
    with Image.open(path) as opened:image=opened.convert('RGB')
    display=image.copy();display.thumbnail((256,128))
    x=column*768+mi*256+(256-display.width)//2
    sheet.paste(display,(x,y+30+(128-display.height)//2))
    rel=path.relative_to(root).as_posix()
    record=dict(case=case['case'],role=role,modality=modality,name=name,dataset_relative_path=rel,
     bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=list(image.size),display_size=list(display.size))
    sources.append(record);originals[rel]=path
 name='sheet_'+str(start//6+1).zfill(3)+'.png';path=output/name;sheet.save(path)
 files[name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=list(sheet.size),cases=[r['case'] for r in rows])
path=output/'originals.zip'
with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
 for rel,source in sorted(originals.items()):z.write(source,rel)
files[path.name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),unique_images=len(originals))
manifest=dict(status='ACTUAL_FIXED_MSVR310_SAVED_ERROR_SHEETS_AND_ORIGINALS_CPU_ONLY',cases=len(cases),files=files,sources=sources,
 display='Each native JPEG fitted within 256x128 preserving aspect ratio, with letterboxing; no crop or enhancement. Original JPEG bytes retained separately in originals.zip.',
 source_order='vis RGB,ni NIR,th TIR',neural_calls=0,model_constructions=0,optimizer_updates=0,Torch_imported=False,training_progress_queries=0)
(output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\\n')
print(json.dumps(manifest))'''
    manifest = json.loads(command(['ssh', *OPTIONS, '2026',
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 " + shlex.quote(PYTHON) + ' -'], input=code))
    assert manifest['neural_calls'] == manifest['model_constructions'] == 0
    assert manifest['cases'] == len(cases) and len(manifest['sources']) == 9 * len(cases)
    LOCAL.mkdir()
    for name, row in manifest['files'].items():
        assert Path(name).name == name
        path = LOCAL / name
        command(['scp', *OPTIONS, '2026:' + REMOTE + '/' + name, str(path)])
        assert path.stat().st_size == row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        row['local'] = str(path)
    cleanup = f'''import hashlib,json
from pathlib import Path
root=Path({REMOTE!r});files={manifest['files']!r}
for name,row in files.items():
 path=root/name;assert path.parent.resolve()==root.resolve()
 assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
for name in files:(root/name).unlink()
(root/'manifest.json').unlink();root.rmdir()
print(json.dumps(dict(removed_verified_duplicates=len(files),original_dataset_untouched=True)))'''
    removed = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=cleanup))
    manifest.update(at=datetime.now().astimezone().isoformat(timespec='seconds'),
        selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cleanup=removed, local_image_directory=str(LOCAL), visual_review='NOT_RUN',
        limits='Fixed label-defined offline sample, not a model selector or new performance measurement. Sheets use explicit aspect-preserving display resizing; original JPEGs retained privately for native inspection. No NN, optimizer, current progress/GPU/weight access or inference about dataset label correctness. Images are not added to the code repository.')
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'], cases=len(cases),
        files=len(manifest['files']), bytes=sum(x['bytes'] for x in manifest['files'].values()),
        local=str(LOCAL), neural_calls=0, visual_review='NOT_RUN'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
