"""Select one existing vehicle recipe; replace only its process-local seed factory."""
import importlib

import identity_coordinate_rgbnt100_expert_seeds as seed_factory

RECIPE_MODULES = dict(AV='run_r201av_RGB_frequency_value_decay', BC='run_r201bc_RGB_reference_interaction')
PARAMETERS = dict(AV=(99585158,1887366), BC=(99617926,1920134))


def recipe_module(key):
    return importlib.import_module(RECIPE_MODULES[key])


def revision(key):
    return (f'RGBNT100 fixed {key} policy, expert seeds42..46 on the same frozen DeMo teacher42. '
        'Only the expert-stage seed changes; same constructor, parameters, FFT, experts, '
        'relations, routing, outlet, optimizer policy, losses, BN2/partial0/two backwards '
        'and full5120 descriptor. Full official8675train/1715query/8575gallery, additional50 '
        'per seed; actual sampler updates and coverage recorded, paired ordinary and axis. '
        'Reuse completed seed42, no seed42 full-training repeat. Normal benchmarkmAP-best '
        'earliest tie, all six same checkpoint; BestSingle under the locked selection rule '
        'and every attempted seed retained. Not independent whole-pipeline repeats, an '
        'untouched test, a new module, calibrated utility or established retrieval gain.')


def build_selected(key,args,cfg,classes,cameras):
    import run_r201g_normal_priority as g
    assert g.coordinate_build is seed_factory.build
    args.expert_recipe = key
    model = recipe_module(key).build(args,cfg,classes,cameras)
    assert (sum(p.numel() for p in model.parameters()),sum(p.numel() for p in model.parameters() if p.requires_grad)) == PARAMETERS[key]
    model.anchor_record.update(method=revision(key),selected_recipe=key,anchor_seed=42,expert_seed=args.seed,
        seed_scope='One fixed teacher42; conditional expert-stage seed only, not a whole-pipeline repeat')
    return model


def configure_runner(key):
    from solver.make_optimizer import make_optimizer as original_make_optimizer
    import run_full_official_frozen_anchor_experiment as runner
    import run_r201g_normal_priority as g
    from original_identity_anchor import assert_anchor_unchanged
    from run_r201ac_RGB_hard_identity import make_loader
    parent = recipe_module(key)
    seed_factory.install_fixed_teacher_expert_seed_builder()
    assert runner.make_optimizer is original_make_optimizer
    runner.make_optimizer = parent.make_optimizer
    runner.configuration = parent.configuration
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = g.record
    runner.make_loader = make_loader
    return runner,parent
