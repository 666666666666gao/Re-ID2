"""Finite F hidden-width configuration on original global AV42, with matched controls."""
import torch
from torch import nn

from axis_collaboration import FrequencyAxisExpert, ModalityAxisExpert
from original_identity_anchor import assert_anchor_unchanged
from rgbnt100_expert_seed_recipe import build_selected, PARAMETERS
from run_r201aq_RGB_attention_decay import TARGETS as ATTENTION_TARGETS
from run_r201av_RGB_frequency_value_decay import TARGETS as VALUE_TARGETS
from solver.make_optimizer import make_optimizer as original_make_optimizer

COUNTS = {64: PARAMETERS['AV'], 128: (99947398, 2249606)}


def revision(rank):
    return (f'RGBNT100 original global AV42, existing F hidden rank{rank} and corresponding '
        'router.to_frequency_query output width. Same expert/module classes, M rank64, '
        '8 heads, number of blocks, original FFT .125/.25, PM/PF/PI and5120 outlet. '
        'AV seven exact no-decay targets and every remaining optimizer group retained; '
        'only their F shape assertions follow the configured rank. Fixed teacher42, '
        'fresh expert42/Adam/additional50, full8675train/1715Q/8575G and same whole '
        'normal mAP-best/earliest-tie checkpoint. Both ordinary and axis use same width. '
        'PaperCLIP acceptance87.2mAP/98.6R1, not our reproduction. A finite internal '
        'capacity test, not a new module, confirmed bottleneck, equal-compute or gain claim.')


def build_rank(rank, args, cfg, classes, cameras):
    assert rank in COUNTS and args.dataset == 'RGBNT100' and args.seed == 42
    assert args.freeze_identity_encoder == 1 and classes == 50
    model = build_selected('AV', args, cfg, classes, cameras)
    assert model.modality_expert.read.__func__ is ModalityAxisExpert.read
    assert model.frequency_expert.prepare.__func__ is FrequencyAxisExpert.prepare
    assert model.frequency_expert.read.__func__ is FrequencyAxisExpert.read
    if rank == 128:
        expert = model.frequency_expert
        pool, scorer = expert.pool, expert.pool_score
        condition = model.router.to_frequency_query
        device = expert.source.device
        # These two CPU constructors draw extra initialization values. Keep the
        # established non-F initialization and subsequent CPU random stream intact.
        with torch.random.fork_rng(devices=[]):
            expanded = FrequencyAxisExpert(model.feat_dim, rank).to(device)
            for name in ('values', 'source', 'query', 'attention', 'cross_band', 'output'):
                setattr(expert, name, getattr(expanded, name))
            condition[1] = nn.Linear(model.feat_dim, rank).to(device)
        assert model.frequency_expert is expert and expert.pool is pool and expert.pool_score is scorer
        assert model.router.to_frequency_query is condition
    assert model.frequency_expert.query.shape == (3, rank)
    assert model.router.to_frequency_query[1].out_features == rank
    assert model.modality_expert.query.shape == (7, 64)
    assert (sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)) == COUNTS[rank]
    assert model.bands.masks.sum((1, 2)).tolist() == [7, 18, 103]
    args.frequency_rank = model.frequency_rank = rank
    model.anchor_record.update(method=revision(rank), frequency_hidden_width=dict(
        rank=rank, reference_rank=64, heads=8, parameter_delta=COUNTS[rank][0]-COUNTS[64][0],
        condition_output_rank=rank, M_rank=64, PM_PF_PI_rank=64, original_global_readers=True,
        original_cutoffs=[.125, .25], descriptor_dim=5120))
    assert_anchor_unchanged(model)
    return model


def make_optimizer(cfg, model, center):
    assert cfg.SOLVER.OPTIMIZER_NAME == 'Adam' and cfg.SOLVER.WEIGHT_DECAY == 1e-4
    optimizer, center_optimizer = original_make_optimizer(cfg, model, center)
    named = dict(model.named_parameters())
    targets = tuple(ATTENTION_TARGETS)+tuple(VALUE_TARGETS)
    ids = {id(named[n]): n for n in targets}
    assert len(ids) == 7 and all(named[n].requires_grad for n in targets)
    rank = model.frequency_rank
    assert named[ATTENTION_TARGETS[0]].shape == (192, 64)
    assert all(named[n].shape == (3*rank, rank) for n in ATTENTION_TARGETS[1:])
    assert all(named[n].shape == (rank, 512) for n in VALUE_TARGETS)
    changed = []
    for group in optimizer.param_groups:
        assert len(group['params']) == 1
        parameter = group['params'][0]
        if id(parameter) in ids:
            assert group['weight_decay'] == 1e-4
            group['weight_decay'] = 0.
            changed.append(ids[id(parameter)])
    assert set(changed) == set(targets)
    model.attention_decay_parameters = [named[n] for n in ATTENTION_TARGETS]
    model.frequency_value_decay_parameters = [named[n] for n in VALUE_TARGETS]
    return optimizer, center_optimizer
