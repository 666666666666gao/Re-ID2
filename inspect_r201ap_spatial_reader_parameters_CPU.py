"""Inspect existing AL query/embedding/input projection tensors, using verified D bytes."""
from array import array
from pathlib import Path
import hashlib,io,json,math,sys,zipfile
from inspect_r201ap_selected_attention_QKV_CPU import MetadataReader,summary

P=Path(__file__).resolve().parent
OUTPUT=P/'results/r201ap_AL_spatial_attention_20261013/analysis/selected_spatial_reader_parameters_CPU.json'


def main():
    receipt=P/'results/preflight/r201al_closed_best_weights_D_only_actual_20261013.json'
    saved=json.loads(receipt.read_text('utf-8-sig'))['files']['RGBNT100_r201al_axis_shared_s42']
    path=Path(saved['local']);assert path.stat().st_size==saved['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==saved['sha256']
    names=['modality_expert.'+k for k in ('source','query','band','value.0.weight','value.0.bias','value.1.weight','value.1.bias')]
    names+=['frequency_expert.'+k for k in ('source','query')]
    names+=['frequency_expert.values.'+str(i)+'.'+k for i in range(3) for k in ('0.weight','0.bias','1.weight','1.bias')]
    assert len(names)==21 and sys.byteorder=='little';rows={}
    with zipfile.ZipFile(path) as archive:
        pickles=[n for n in archive.namelist() if n.endswith('/data.pkl')];assert len(pickles)==1
        prefix=pickles[0].rsplit('/',1)[0];assert archive.read(prefix+'/byteorder')==b'little'
        tensors=MetadataReader(io.BytesIO(archive.read(pickles[0]))).load()
        for name in names:
            t=tensors[name];shape=t['shape'];assert t['storage']['dtype']=='FloatStorage' and t['offset']==0
            assert t['stride']==((shape[1],1) if len(shape)==2 else (1,))
            values=array('f');values.frombytes(archive.read(prefix+'/data/'+t['storage']['key']))
            assert len(values)==math.prod(shape) and all(math.isfinite(v) for v in values)
            rows[name]=dict(shape=list(shape),**summary(values))
    sources={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ('inspect_r201ap_spatial_reader_parameters_CPU.py','inspect_r201ap_selected_attention_QKV_CPU.py')}
    result=dict(status='ACTUAL_AL_SELECTED_SPATIAL_READER_PARAMETERS_ARCHIVED_BYTES_CPU_INSPECTION',
        checkpoint=dict(bytes=saved['bytes'],sha256=saved['sha256'],selected_epoch=20),source_sha256=sources,
        parameters=rows,model_forwards=0,optimizer_updates=0,GPU_calls=0,new_checkpoint_writes=0,
        limits='Existing selected AL snapshot only, 21 query/source/band and input LayerNorm/linear tensors. Magnitudes are not spatial activation variance, matrix rank, gradient trajectories or proof of missing identity information. AQ live sources/optimizer/processes are not queried or changed. Float32 bytes accumulated as Python double; standard deviation uses sample denominator.')
    assert not OUTPUT.exists();OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],norms={n:r['norm'] for n,r in rows.items()})),flush=True)


if __name__=='__main__':main()
