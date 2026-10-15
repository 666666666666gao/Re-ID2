"""Full-official global AV42 training with one fixed internal cutoff pair."""
import argparse
import sys

from rgbnt100_expert_seed_recipe import configure_runner
from rgbnt100_frequency_band_ranges import build_ranges,revision


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--band-low-cutoff',type=float,required=True)
    parser.add_argument('--band-high-cutoff',type=float,required=True)
    selected,remaining=parser.parse_known_args()
    low,high=selected.band_low_cutoff,selected.band_high_cutoff
    assert (low,high)==(.25,.375)
    sys.argv=[sys.argv[0],*remaining]
    runner,parent=configure_runner('AV')

    def build(args,cfg,classes,cameras):
        return build_ranges(low,high,args,cfg,classes,cameras)

    def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
        model.band_range_reconstruction_errors.clear()
        detail=parent.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
        detail['frequency_band_ranges']=dict(model.anchor_record['frequency_band_ranges'])
        if model.normal_priority_smoke:
            errors=model.band_range_reconstruction_errors
            assert errors and max(errors)<1e-5
            detail['frequency_band_reconstruction']=dict(actual_forward_calls=len(errors),maximum_absolute_error=max(errors))
        return detail

    runner.main(builder=build,update=step,variants=('frequency_shared','axis_shared'),method_revision=revision(low,high),
        partial_training='Unchanged global AV42 BN2/partial0/two backwards; no missing-specialist training.')


if __name__=='__main__':main()
