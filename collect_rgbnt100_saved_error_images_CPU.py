"""Collect private contact sheets from fixed completed-cache error pairs; no NN."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from collect_results import OPTIONS

PROJECT = Path(__file__).resolve().parent
ANALYSIS = PROJECT / 'results/rgbnt100_saved_error_cases_CPU_20261015'
LOCAL = Path('C:/Users/gb/.codex_tmp/RGBNT100_saved_error_images_CPU_20261015')
REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis/diagnostics/rgbnt100_saved_error_images_CPU_20261015'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'


def command(argv, **kwargs):
    result = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', timeout=300, **kwargs)
    if result.returncode:
        print(result.stdout, flush=True)
        print(result.stderr, file=sys.stderr, flush=True)
        result.check_returncode()
    return result.stdout


def main():
    output = PROJECT / 'results/preflight/rgbnt100_saved_error_images_CPU_collection_actual_20261015.json'
    assert not output.exists() and not LOCAL.exists()
    analysis = json.loads((ANALYSIS / 'result.json').read_text('utf-8'))
    assert analysis['status'] == 'ACTUAL_SAVED_AV00_11_COMPLETE_NEAREST_MATCH_CPU_ANALYSIS_NO_NN'
    selection_path = ANALYSIS / 'selected_visual_cases.json'
    cases = json.loads(selection_path.read_text('utf-8'))['cases']
    code = f'''import hashlib,json,os,sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert 'torch' not in sys.modules
root=Path('/data/gaob/Re-ID/dataset/RGBNT100/rgbir')
output=Path({REMOTE!r});assert not output.exists();output.mkdir(parents=True)
cases={cases!r}
font=ImageFont.load_default()
files={{}};sources=[]
for start in range(0,len(cases),6):
 rows=cases[start:start+6]
 sheet=Image.new('RGB',(2304,48+160*len(rows)),(20,24,30));draw=ImageDraw.Draw(sheet)
 for column,title in enumerate(('QUERY   RGB | NIR | TIR','WRONG TOP1   RGB | NIR | TIR','CORRECT ID   RGB | NIR | TIR')):
  draw.text((column*768+8,12),title,fill='white',font=font)
 for row_index,case in enumerate(rows):
  y=48+row_index*160
  for column,role in enumerate(('query','wrong','correct')):
   name=case[role+'_name'];assert Path(name).name==name
   path=root/('query' if role=='query' else 'bounding_box_test')/name
   with Image.open(path) as opened:image=opened.convert('RGB')
   assert image.size==(768,128)
   sheet.paste(image,(column*768,y+30))
   color='white' if role=='query' else '#ff8c8c' if role=='wrong' else '#8cffb0'
   draw.text((column*768+8,y+5),'case'+str(case['case'])+' '+case['cohort']+' '+role+' '+name,fill=color,font=font)
   sources.append(dict(case=case['case'],role=role,name=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=list(image.size)))
  # Thin dividers mark modalities without changing any image pixel.
  for x in (768,1536):draw.line((x,y,x,y+28),fill=(80,85,92))
 name='sheet_'+str(start//6+1).zfill(3)+'.png';path=output/name;sheet.save(path)
 files[name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=list(sheet.size),cases=[row['case'] for row in rows])
manifest=dict(status='ACTUAL_FIXED_SAVED_ERROR_CASE_CONTACT_SHEETS_CPU_ONLY',cases=len(cases),files=files,sources=sources,
 raw_pixels_preserved_without_resizing=True,source_order='RGB,NIR,TIR montage, as current Triplets reader',
 neural_calls=0,model_constructions=0,optimizer_updates=0,Torch_imported=False,training_progress_queries=0)
(output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\\n')
print(json.dumps(manifest))'''
    manifest = json.loads(command(['ssh', *OPTIONS, '2026', "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 " + shlex.quote(PYTHON) + ' -'], input=code))
    assert manifest['status'] == 'ACTUAL_FIXED_SAVED_ERROR_CASE_CONTACT_SHEETS_CPU_ONLY'
    assert manifest['neural_calls'] == manifest['model_constructions'] == 0
    assert manifest['cases'] == len(cases) and len(manifest['sources']) == 3 * len(cases)
    LOCAL.mkdir()
    for name, row in manifest['files'].items():
        assert Path(name).name == name and name.endswith('.png')
        path = LOCAL / name
        command(['scp', *OPTIONS, '2026:' + REMOTE + '/' + name, str(path)])
        assert path.stat().st_size == row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        row['local'] = str(path)
    cleanup = f'''import hashlib,json
from pathlib import Path
root=Path({REMOTE!r});files={manifest['files']!r}
for name,row in files.items():
 path=root/name;assert path.resolve().is_relative_to(root.resolve())
 assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
for name in files:(root/name).unlink()
(root/'manifest.json').unlink();root.rmdir()
print(json.dumps(dict(removed_duplicate_sheets=len(files),original_dataset_untouched=True)))'''
    removed = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=cleanup))
    manifest.update(at=datetime.now().astimezone().isoformat(timespec='seconds'),
        selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cleanup=removed,local_image_directory=str(LOCAL),
        visual_review='NOT_RUN',
        limits='Fixed diagnostic sample, one lowest query index per identity within each non-correct cohort. Raw source montages copied unchanged into private sheets. No image-derived fit, inference, score, new experiment, current BC progress/GPU/weight access or generalization claim. Images are kept locally, not added to the code repository.')
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'],cases=len(cases),sheets=len(manifest['files']),
        local=str(LOCAL),neural_calls=0,visual_review='NOT_RUN'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
