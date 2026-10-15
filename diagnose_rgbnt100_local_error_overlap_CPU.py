"""Count fixed-checkpoint rescue overlap; GT oracle is diagnosis, never deployment."""
import csv
import json
from pathlib import Path

P=Path(__file__).resolve().parent


def read(path):return list(csv.DictReader(path.open(encoding='utf-8',newline='')))


def main():
    base=P/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42/best_per_query.csv'
    global_path=P/'results/r201av_RGB_frequency_value_decay_20261014/RGBNT100/training/RGBNT100_r201av_axis_shared_s42/best_per_query.csv'
    root=P/'results/rgbnt100_F_local_grid2_AV_20261016/RGBNT100'
    local_path=root/'training/RGBNT100_flocal2_AV_axis_shared_s42/best_per_query.csv'
    states_path=root/'states/RGBNT100_flocal2_AV_axis_shared_s42'
    before,global_rows,local_rows=read(base),read(global_path),read(local_path)
    states={key:read(states_path/('state_'+key+'.csv')) for key in ('00','10','01','11')}
    keys=('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')
    assert len(before)==1715
    for rows in (global_rows,local_rows,*states.values()):
        assert len(rows)==1715 and all(tuple(x[k] for k in keys)==tuple(y[k] for k in keys) for x,y in zip(before,rows))
    errors={i for i,row in enumerate(before) if not int(row['Rank-1'])}
    assert len(errors)==58
    rescued=lambda rows:{i for i in errors if int(rows[i]['Rank-1'])}
    harmed=lambda rows:{i for i in range(1715) if i not in errors and not int(rows[i]['Rank-1'])}
    global_rescue,local_rescue=rescued(global_rows),rescued(local_rows)
    state_rescue={key:rescued(rows) for key,rows in states.items()}
    union=set.union(*state_rescue.values())
    result=dict(status='ACTUAL_F_LOCAL_SAVED_FULL_QUERY_RESCUE_OVERLAP_CPU',queries=1715,original_errors=58,
        global_AV_rescued=len(global_rescue),global_AV_harmed=len(harmed(global_rows)),
        local_rescued=len(local_rescue),local_harmed=len(harmed(local_rows)),
        common_rescues=len(global_rescue&local_rescue),local_unique_rescues=len(local_rescue-global_rescue),global_unique_rescues=len(global_rescue-local_rescue),
        local_unique_rescue_rows=[dict(query_index=i,name=before[i]['name'],identity=before[i]['identity']) for i in sorted(local_rescue-global_rescue)],
        local_four_state_rescue_counts={key:len(value) for key,value in state_rescue.items()},
        local_four_state_GT_oracle_rescue_count=len(union),local_four_state_GT_oracle_Rank1=100*(1715-len(errors)+len(union))/1715,
        target_Rank1=97.61807580174927,target_net_correct_increase=18,new_NN=0,
        limits='All1715 saved queries from these particular fixed checkpoints, using true labels only for offline diagnosis. The oracle assumes a perfect GT-based selector including base00; it is not deployable, a selected new model or an upper bound for future representations. This does not prove a causal reason for failure or justify more seeds.')
    out=P/'results/preflight/rgbnt100_F_local_saved_error_overlap_CPU_actual_20261016.json'
    assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
