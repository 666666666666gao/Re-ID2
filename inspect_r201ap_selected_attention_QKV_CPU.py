"""Read exact archived AL tensor bytes and report Q/K/V block magnitudes; no Torch."""
from array import array
from collections import OrderedDict
from pathlib import Path
import hashlib,io,json,math,pickle,statistics,sys,zipfile


def tensor_metadata(storage,offset,shape,stride,requires_grad,hooks):
    return dict(storage=storage,offset=offset,shape=shape,stride=stride)


class MetadataReader(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('collections','OrderedDict'):OrderedDict,('torch._utils','_rebuild_tensor_v2'):tensor_metadata,
            ('torch','FloatStorage'):'FloatStorage',('torch','LongStorage'):'LongStorage',('torch','BoolStorage'):'BoolStorage'}
        return allowed[(module,name)]

    def persistent_load(self,value):
        assert len(value)==5 and value[0]=='storage'
        return dict(dtype=value[1],key=value[2],device=value[3],elements=value[4])


def summary(values):
    return dict(norm=math.sqrt(math.fsum(v*v for v in values)),absolute_max=max(abs(v) for v in values),
        standard_deviation=statistics.stdev(values),elements=len(values),nonzero=sum(v!=0 for v in values))


def main():
    receipt=json.loads(Path('results/preflight/r201al_closed_best_weights_D_only_actual_20261013.json').read_text('utf-8-sig'))
    saved=receipt['files']['RGBNT100_r201al_axis_shared_s42'];path=Path(saved['local'])
    assert path.stat().st_size==saved['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==saved['sha256']
    assert sys.byteorder=='little';rows={}
    with zipfile.ZipFile(path) as archive:
        pickles=[n for n in archive.namelist() if n.endswith('/data.pkl')];assert len(pickles)==1
        prefix=pickles[0].rsplit('/',1)[0];assert archive.read(prefix+'/byteorder')==b'little'
        tensors=MetadataReader(io.BytesIO(archive.read(pickles[0]))).load()
        names=[n for n in tensors if n.endswith(('in_proj_weight','in_proj_bias')) and n.startswith(('generalFusion.','modality_expert.','frequency_expert.'))]
        for name in names:
            t=tensors[name];shape=t['shape'];assert t['storage']['dtype']=='FloatStorage'
            assert t['offset']==0 and t['stride']==((shape[1],1) if len(shape)==2 else (1,))
            values=array('f');values.frombytes(archive.read(prefix+'/data/'+t['storage']['key']))
            assert len(values)==math.prod(shape) and shape[0]%3==0
            chunk=len(values)//3
            for index,label in enumerate(('Q','K','V')):rows[name+'_'+label]=summary(values[index*chunk:(index+1)*chunk])
    out=Path('results/r201ap_AL_spatial_attention_20261013/analysis/selected_QKV_parameter_blocks_CPU.json')
    assert not out.exists()
    result=dict(status='ACTUAL_AL_SELECTED_QKV_ARCHIVED_BYTES_CPU_INSPECTION',selected_epoch=20,
        archived_checkpoint_size_and_sha_exact=True,checkpoint_sha256=saved['sha256'],projection_blocks=rows,
        model_forwards=0,optimizer_updates=0,GPU_calls=0,new_checkpoint_writes=0,
        estimator='Float32 checkpoint bytes accumulated as Python double; standard_deviation uses sample denominator.',
        limits='Selected snapshot magnitudes only, not training trajectories, optimizer-versus-task gradient attribution, matrix rank or proof that patch features lack spatial detail. Runtime spatial distributions are measured separately by AP full10290 exact-output forward.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in rows.items() if k.startswith(('modality_expert','frequency_expert')) and 'weight' in k},indent=2),flush=True)


if __name__=='__main__':main()
