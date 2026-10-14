"""Summarize completed AV/AX F reading-scale and gradient traces; no NN or live logs."""
import csv,json,math
from pathlib import Path

PROJECT=Path(__file__).resolve().parent
TAGS={'AV':'r201av_RGB_frequency_value_decay_20261014',
      'AX':'r201ax_RGB_frequency_query_normalization_20261014'}
METRICS=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    roots={stage:PROJECT/'results'/tag/'RGBNT100' for stage,tag in TAGS.items()}
    # This is deliberately a terminal-only reducer, including the original AX controller.
    for root in roots.values():
        controller=json.loads((root/'controller_result.json').read_text('utf-8-sig'))
        assert controller['status']=='COMPLETE' and controller['successful_updates']==13126
        assert controller['native_updates']==6 and controller['states_cases']==8
    output=PROJECT/'results'/TAGS['AX']/'F_query_trace_CPU'
    output.mkdir(exist_ok=False)
    traces=[];endpoints=[];counts={}
    for stage,root in roots.items():
        for variant in ('frequency_shared','axis_shared'):
            name='RGBNT100_r201'+stage.lower()+'_'+variant+'_s42'
            folder=root/'training'/name
            result=json.loads((folder/'result.json').read_text('utf-8-sig'))
            assert result['status']=='COMPLETE' and result['epochs']==50
            assert result['optimizer_steps']==result['steps']==6563
            assert result['training_heldout_identities']==result['amp_skipped_steps']==0
            epochs=read_csv(folder/'epochs.csv')
            assert [int(row['epoch']) for row in epochs]==list(range(1,51))
            assert int(max(epochs,key=lambda row:float(row['mAP']))['epoch'])==result['best']['epoch']
            batches={i:[] for i in range(1,51)};steps=0
            with (folder/'batch_orders.jsonl').open(encoding='utf-8-sig') as stream:
                for line in stream:
                    row=json.loads(line);steps+=1
                    assert row['optimizer_updated'] and row['loss']==row['full_loss']
                    assert row['partial_loss_effective']==0
                    decay=row['attention_projection_decay']
                    assert len(decay['unscaled_task_gradient_norm'])==15
                    for band in range(3):
                        # Existing target order: M, F-low, F-mid, F-high, cross-band; Q/K/V per target.
                        offset=3*(band+1)
                        values={key:float(decay['unscaled_task_gradient_norm'][offset+j])
                                for j,key in enumerate(('Q_task_gradient','K_task_gradient','V_task_gradient'))}
                        values['loss']=float(row['loss'])
                        values['M_projected_Q_pre_norm']=float(row['modality_projected_query_normalization']['pre_norm_mean'])
                        if stage=='AX':
                            policy=row['frequency_projected_query_normalization']
                            assert policy['projected_keys_values_unchanged'] and policy['cross_band_unchanged']
                            assert policy['parameters_added']==policy['extra_forwards']==policy['extra_losses']==policy['extra_updates']==0
                            q=policy['bands'][band]
                            assert q['attention_calls']==2 and q['pre_norm_min']>0
                            assert abs(q['post_norm_min']-math.sqrt(8))<1e-5
                            assert abs(q['post_norm_max']-math.sqrt(8))<1e-5
                            values.update(F_projected_Q_pre_norm=float(q['pre_norm_mean']),
                                          F_projected_Q_post_min_mean=float(q['post_norm_min']),
                                          F_projected_Q_post_max_mean=float(q['post_norm_max']))
                        assert all(math.isfinite(v) for v in values.values())
                        batches[row['epoch']].append((band,values))
            assert steps==6563;counts[stage+'_'+variant]=steps
            for epoch in epochs:
                number=int(epoch['epoch'])
                for band in range(3):
                    members=[values for index,values in batches[number] if index==band]
                    assert members
                    means={key:math.fsum(v[key] for v in members)/len(members) for key in members[0]}
                    assert abs(means['loss']-float(epoch['loss']))<1e-8
                    # No value is invented for AV: its F Q norms were not recorded during training.
                    traces.append(dict(stage=stage,variant=variant,epoch=number,band=band,
                        batches=len(members),**means,**{k:float(epoch[k]) for k in METRICS}))
            endpoints.append(dict(stage=stage,variant=variant,selected_epoch=result['best']['epoch'],
                best={k:result['full_metrics'][k] for k in METRICS},
                last={k:float(epochs[-1][k]) for k in METRICS},
                mAP_best_minus_last=result['full_metrics']['mAP']-float(epochs[-1]['mAP'])))
    keys=list(dict.fromkeys(key for row in traces for key in row))
    with (output/'epoch_band_traces.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(traces)
    proof=dict(status='ACTUAL_CLOSED_AV_AX_F_QUERY_TRAINING_TRACE_CPU_COMPLETE',
        rows=len(traces),records=counts,endpoints=endpoints,NN_calls=0,optimizer_updates=0,
        limits='Training-batch gradient and FP32 pre/post Q scale diagnostics only. AV lacks recorded F Q norms, so those CSV cells remain absent. Equal epoch gradient magnitudes do not imply identical examples of hard mining or causal identity usefulness. Benchmark-selected best/last remain same-checkpoint metrics; no missing or seed claim.')
    (output/'result.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(status=proof['status'],rows=len(traces),NN_calls=0)))


if __name__=='__main__':main()
