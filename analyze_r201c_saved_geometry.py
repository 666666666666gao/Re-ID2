"""Read the observed float32 C-order NPY arrays; no NumPy/install/NN needed."""
import ast
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import zipfile


def floats(archive, key):
    data = archive.read(key + '.npy')
    assert sys.byteorder == 'little' and data[:8] == b'\x93NUMPY\x01\x00'
    length = struct.unpack('<H', data[8:10])[0]
    header = ast.literal_eval(data[10:10+length].decode('latin1'))
    assert header == dict(descr='<f4', fortran_order=False, shape=(836,5120))
    values = memoryview(data)[10+length:].cast('f')
    assert len(values) == 836*5120
    return values


def main():
    project = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
    root = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')
    base_path = root / 'rgbnt201_identity_outlet_native_20261005/axis_shared/bypass_0/readout_arrays.npz'
    out = project / 'results/r201d_preparation_20261006/saved_best_geometry'
    out.mkdir(exist_ok=False)
    rows = []
    with zipfile.ZipFile(base_path) as base_archive:
        for variant in ('frequency_shared','axis_shared'):
            path = root / 'rgbnt201_identity_outlet_r201c_20261005/training' / ('RGBNT201_identity_' + variant + '_narrow_s42') / 'best_official_arrays.npz'
            with zipfile.ZipFile(path) as saved:
                for key in ('query_ids','query_cameras','query_scenes','query_names','gallery_ids','gallery_cameras','gallery_scenes','gallery_names'):
                    assert base_archive.read(key+'.npy') == saved.read(key+'.npy')
                for role in ('query','gallery'):
                    base = floats(base_archive,'private_'+role+'_features')
                    fused = floats(saved,role+'_features')
                    for index in range(836):
                        b = base[index*5120:(index+1)*5120]
                        z = fused[index*5120:(index+1)*5120]
                        assert abs(math.fsum(x*x for x in b)-1) < 1e-5
                        assert abs(math.fsum(x*x for x in z)-1) < 1e-5
                        bg = math.fsum(x*x for x in b[:1536]); zg = math.fsum(x*x for x in z[:1536])
                        assert bg > 0 and zg > 0
                        scale = math.sqrt(bg/zg)
                        residual = [scale*y-x for x,y in zip(b,z)]
                        global_error = max(abs(x) for x in residual[:1536])
                        assert global_error < 2e-6
                        energy = math.fsum(x*x for x in residual)
                        radial = math.fsum(x*y for x,y in zip(b,residual))
                        cosine = math.fsum(x*y for x,y in zip(b,z))
                        assert energy > 0
                        rows.append(dict(variant=variant,role=role,index=index,
                            relative_raw_correction_norm=math.sqrt(energy), radial_energy_fraction=radial*radial/energy,
                            tangent_energy_fraction=1-radial*radial/energy,
                            direction_angle_degrees=math.degrees(math.acos(min(1,max(-1,cosine)))),
                            fused_to_base_raw_norm_ratio=scale,unchanged_global_max_error=global_error))
    assert len(rows)==3344
    with (out/'per_sample.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='ACTUAL_SELECTED_R201C_NORMAL_CORRECTION_GEOMETRY',observed_at=datetime.now().isoformat(timespec='seconds'),
        samples=3344,source=str(base_path),new_neural_calls=0,new_optimizer_updates=0,groups={})
    fields=('relative_raw_correction_norm','radial_energy_fraction','tangent_energy_fraction','direction_angle_degrees','fused_to_base_raw_norm_ratio','unchanged_global_max_error')
    for variant in ('frequency_shared','axis_shared'):
        for role in ('query','gallery'):
            subset=[r for r in rows if r['variant']==variant and r['role']==role]
            result['groups'][variant+'/'+role]={k:dict(mean=statistics.mean(r[k] for r in subset),median=statistics.median(r[k] for r in subset),maximum=max(r[k] for r in subset)) for k in fields}
    result['limits']='Selected early best only. Identity base is fixed originalE28 from actual native; unchanged globals verify the relative raw reconstruction. No late50 checkpoint retained, so this cannot measure late residual growth or establish the collapse cause. This is not expert orthogonality, causal utility, or multiseed evidence.'
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
