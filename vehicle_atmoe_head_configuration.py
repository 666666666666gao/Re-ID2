"""Configure the existing DeMo ATMoE partition count for vehicle datasets."""
from run_experiment import configuration as original_configuration


def configuration(args, heads):
    assert (args.dataset, args.variant, heads) in (
        ('RGBNT100', 'demo', 8), ('RGBNT100', 'demo', 32),
        ('MSVR310', 'demo_shared', 8), ('MSVR310', 'demo_shared', 16))
    assert args.seed == 42
    cfg = original_configuration(args).clone()
    assert cfg.MODEL.HEAD == 8 and cfg.MODEL.HDM and cfg.MODEL.ATM
    cfg.defrost()
    cfg.MODEL.HEAD = heads
    cfg.freeze()
    args.atmoe_heads = heads
    return cfg


def verify_partition(model, heads):
    assert model.feat_dim == 512 and model.head == heads
    moe = model.generalFusion.moe
    assert moe.head == heads and moe.head_dim == 512 // heads
    assert len(moe.experts) == heads
    assert moe.gating_network.gate.num_heads == heads
    assert all(len(part.expertHead) == 7 for part in moe.experts)
    assert all(expert.mlp[0].in_features == expert.mlp[0].out_features == 512 // heads
               for part in moe.experts for expert in part.expertHead)
    # HDM's seven source-relation readers use a separate, unchanged head count.
    assert all(getattr(model.generalFusion, name).num_heads == 8
               for name in ('r', 'n', 't', 'rn', 'rt', 'nt', 'rnt'))
