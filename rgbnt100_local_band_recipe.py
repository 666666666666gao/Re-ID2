"""One existing RGBNT100 recipe with only F's internal spatial reading changed."""
from local_band_evidence_reading import install_region_reading
from rgbnt100_expert_seed_recipe import build_selected, PARAMETERS


def revision(key, grid):
    return (f'RGBNT100 {key} parent, fixed teacher42/expert42, F-only grid{grid}. '
        'Structured F retains same-source regions through existing cross-band attention '
        'and nonlinear output; ordinary pools mixed sources within each image region. '
        'M, FFT, conditional round/router, PM/PF/PI, gates, descriptor5120, loss/optimizer '
        'and BN2/partial0/two backwards unchanged. Zero added learned parameters. '
        'Full official8675train/1715query/8575gallery, additional50, benchmarkmAP-best '
        'earliest tie and all six from one checkpoint. Fixed rectangles do not establish '
        'semantic part alignment. Same parameter count does not establish equal runtime '
        'cost. Source preparation is not model parity, native admission or a result.')


def build_local(key, grid, args, cfg, classes, cameras):
    assert args.dataset == 'RGBNT100' and args.seed == 42
    assert args.freeze_identity_encoder == 1 and grid in (1, 2)
    assert tuple(cfg.INPUT.SIZE_TRAIN) == tuple(cfg.INPUT.SIZE_TEST) == (128, 256)
    assert tuple(cfg.MODEL.STRIDE_SIZE) == (16, 16)
    model = build_selected(key, args, cfg, classes, cameras)
    args.local_parent_recipe, args.local_reading_grid = key, grid
    model = install_region_reading(model, 8, 16, grid)
    assert (sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad)) == PARAMETERS[key]
    model.anchor_record.update(method=revision(key, grid),
        local_parent_recipe=key,
        seed_scope='Fixed teacher42 and expert42; local reading comparison, no seed sweep',
        construction_scope='Reuse reviewed fixed-teacher seed factory; this is not an N5 run or N5 selection')
    return model
