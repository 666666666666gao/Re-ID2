"""Describe existing FFT bin allocation and saved full AV42 spatial statistics."""
import csv
import hashlib
import json
import math
from pathlib import Path

P=Path(__file__).resolve().parent


def masks(low,high):
    rows=[[],[],[]]
    for y in range(8):
        for x in range(16):
            squared=(min(y,8-y)/8)**2+(min(x,16-x)/16)**2
            band=0 if squared<=low**2 else 1 if squared<=high**2 else 2
            for k in range(3):
                rows[k].append(k==band)
    assert all(sum(row[i] for row in rows)==1 for i in range(128))
    assert all(row[y*16+x]==row[((-y)%8)*16+((-x)%16)] for row in rows for y in range(8) for x in range(16))
    return dict(low=low,high=high,bins=[sum(row) for row in rows],fractions=[sum(row)/128 for row in rows],complete_disjoint_conjugate_symmetric=True)


def main():
    base=P/'results/r201aw_AV_frequency_outlets_20261014'
    path=base/'RGBNT100/full/RGBNT100_r201av_axis_shared_s42/spatial_attention_records.csv'
    rows=list(csv.DictReader(path.open(encoding='utf-8',newline='')))
    assert len(rows)==10290 and len(rows[0])==282
    assert {side:sum(r['side']==side for r in rows) for side in ('query','gallery')}==dict(query=1715,gallery=8575)
    values=[]
    for side in ('query','gallery'):
        members=[r for r in rows if r['side']==side]
        for source in range(3):
            for band in range(3):
                prefix='F_source'+str(source)+'_band'+str(band)+'_'
                fields=('input_mean_square','input_spatial_variance','after_LN_spatial_variance','after_GELU_spatial_variance')
                selected={field:[float(r[prefix+field]) for r in members] for field in fields}
                assert all(math.isfinite(v) for series in selected.values() for v in series)
                assert min(selected['input_mean_square'])>0
                values.append(dict(side=side,source=source,band=band,records=len(members),
                    **{field:math.fsum(series)/len(members) for field,series in selected.items()},
                    mean_input_spatial_fraction=math.fsum(v/e for v,e in zip(selected['input_spatial_variance'],selected['input_mean_square']))/len(members)))
    result=dict(status='ACTUAL_SAVED_AV42_FULL_QG_BAND_RANGE_CPU_DIAGNOSTIC',
        source=str(path.relative_to(P)),source_bytes=path.stat().st_size,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rows=10290,query_records=1715,gallery_records=8575,columns=282,spatial_measure_columns=275,
        original=masks(.125,.25),single_candidate_not_run=masks(.25,.375),source_band_statistics=values,
        new_NN=0,new_training=0,current_job_queries=0,
        interpretation='The existing vehicle8x16 FFT grid allocates7/18/103 frequency positions to L/M/H. A single alternative25/30/73 would preserve three bands and all module roles while changing only cutoff parameters. This does not establish equal energy, unequal usefulness, an implementation bug, the cause of poor retrieval, or a future gain. Saved spatial statistics belong to existing full AV42, not new F-local training.',
        future_control='If pursued, return to unchanged global AV42 readers and test only cutoff.25/.375 with matched ordinary/axis, same teacher42/expert42/full data/extra50. Do not retain failed localgrid2, change losses/depth/seed concurrently or claim cutoff tuning as a novel module.')
    out=P/'results/preflight/rgbnt100_band_range_saved_AV42_CPU_actual_20261016.json'
    assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ('status','rows','columns','original','single_candidate_not_run','new_NN')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
