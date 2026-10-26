"""Existing AV controls on one new B128/head8 identity reference and its train-only bank."""
import hashlib,json
from pathlib import Path

import run_r201av_RGB_frequency_value_decay as recipe
import run_r201ac_RGB_hard_identity as sampling

configuration=recipe.configuration
VARIANTS=recipe.VARIANTS
BANK_PATH=Path('runs/rgbnt100_B128_training_neighbors_20261017/result.json')
REVISION=('Existing AV architecture, experts, FFT, one conditional interaction, routingT2, '
    'PM/PF/PI, cap.10, auxiliary.001, mainCE.25, hinge.6, BN2/partial0/two backwards and '
    'P8K8/B64 expert50 are retained. Only identity reference is the completed full50 '
    'public-CLIP B128/head8 baseline, and its mandatory train-only centroid bank is '
    'rebuilt against exactly that weight. This also changes the teacher-derived hard '
    'identity arrangement; do not call comparison against oldAV a pure single-factor '
    'network ablation. Matched ordinary/axis share teacher, bank, newheads42 and '
    'newAdam/extra50. No new module/parameter/loss, seed search,201NN, missing or49. '
    'PaperRGBNT10087.2/98.6 acceptance, allsix from own earliestnormalmAP-best; '
    'officialbenchmark selected teacher/config/epoch, not an untouched test.')


def selected_bank_build(args,cfg,classes,cameras):
    # AC binds one literal old bank to its teacher. Keep its sampler/update and replace
    # only that build-time bank source; this module runs in an isolated process.
    model=sampling.x.build(args,cfg,classes,cameras)
    data=BANK_PATH.read_bytes();bank=json.loads(data)
    assert bank['status']=='COMPLETE_RGBNT100_TRAIN_ONLY_NEIGHBOR_BANK'
    assert bank['train_records']==8675 and bank['train_identities']==classes==50
    assert bank['identity_heads']==8 and bank['identity_training_batch']==128
    assert bank['query_gallery_forwards']==bank['optimizer_updates']==bank['new_weights']==0
    assert bank['original_teacher_state_exact']
    assert hashlib.sha256((Path(args.anchor_run_dir)/'best.pth').read_bytes()).hexdigest()==bank['teacher_sha256']
    assert set(map(int,bank['nearest_identity_labels']))==set(range(50))
    assert all(set(ids)==set(range(50))-{int(pid)} and len(ids)==49 for pid,ids in bank['nearest_identity_labels'].items())
    sampling.BANK=bank
    sampling.experiment_data.RandomIdentitySampler=sampling.HardIdentitySampler
    model.anchor_record['hard_identity_sampling']=dict(bank=str(BANK_PATH),bank_sha256=hashlib.sha256(data).hexdigest(),
        training_only=True,P=8,K=8,top_neighbors=3,maximum_swaps_per_batch=1,parameters_added=0,
        original_sampler_RNG_consumed_first=True,epoch_record_multiset_preserved=True)
    return model


sampling.build=selected_bank_build


def build(args,cfg,classes,cameras):
    anchor=Path(args.anchor_run_dir)
    trained=json.loads((anchor/'result.json').read_text())
    assert anchor.name=='RGBNT100_B128_atmoe8_demo_s42'
    assert trained['arguments']['atmoe_heads']==8 and trained['arguments']['training_batch']==128
    assert trained['steps']==trained['optimizer_steps']==3135 and trained['amp_skipped_steps']==0
    model=recipe.build(args,cfg,classes,cameras)
    model.anchor_record.update(method=REVISION,identity_reference_training_batch=128,
        identity_reference_heads=8,reference_specific_bank=str(BANK_PATH),expert_training_batch=64,
        changes_vs_old_AV='Selected identity weights and their train-only centroid neighbor arrangement; AV recipe unchanged')
    return model


step=recipe.step
make_optimizer=recipe.make_optimizer


if __name__=='__main__':
    import run_full_official_frozen_anchor_experiment as runner
    assert runner.make_optimizer is recipe.parent.parent.original_make_optimizer
    runner.make_optimizer=make_optimizer
    runner.configuration=configuration
    runner.assert_anchor_unchanged=sampling.x.w.assert_anchor_unchanged
    runner.write_json=sampling.x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.make_loader=sampling.make_loader
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Same AV BN2/partial0/two backwards; no missing design. New head8/B128 reference and its explicitly matched train-only bank.')
