"""Initialize the existing final classifier using the immutable teacher TRAIN bank."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.nn import functional as F

BANK_PATH = Path('inputs/r201ba_RGB_training_bank_20261015/training_features.npz')
BANK_SHA = '9d1f92ec7b3755debf2cecee4980a574d70f3b6c1ce37ffc760d0a7197e2cba9'
BANK_BYTES = 167098445


def initialize(model, bank):
    weight = model.fused_classifier.weight
    assert weight.shape == (50, 5120) and weight.dtype == torch.float32
    assert model.fused_classifier.bias is None
    assert BANK_PATH.stat().st_size == BANK_BYTES
    assert hashlib.sha256(BANK_PATH.read_bytes()).hexdigest() == BANK_SHA
    assert bank['train_records'] == 8675 and bank['train_identities'] == 50
    assert bank['query_gallery_forwards'] == bank['optimizer_updates'] == bank['new_weights'] == 0
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    with np.load(BANK_PATH, allow_pickle=False) as raw:
        features, labels = raw['features'], raw['train_ids']
        records = bank['train_records_metadata']
        assert features.shape == (8675, 5120) and features.dtype == np.float32
        assert np.isfinite(features).all()
        assert np.max(np.abs(np.linalg.norm(features, axis=1) - 1)) < 1e-6
        assert raw['train_names'].tolist() == [r['name'] for r in records]
        expected = [bank['training_labels'][str(r['identity'])] for r in records]
        assert np.array_equal(labels, expected) and np.array_equal(np.unique(labels), np.arange(50))
        mean = features.mean(axis=0, dtype=np.float64)
        variance = features.var(axis=0, dtype=np.float64)
        centers = np.stack([features[labels == label].mean(axis=0, dtype=np.float64) for label in range(50)])
    # The final CE reads the existing BN coordinate. Use its existing epsilon;
    # this is deterministic initialization, not a replacement BN or new loss.
    directions = (centers - mean) / np.sqrt(variance + model.fused_neck.eps)
    assert np.isfinite(directions).all()
    assert np.all(np.linalg.norm(directions, axis=1) > 0)
    before = weight.detach().cpu().clone()
    norms = before.norm(dim=1)
    directions = torch.from_numpy(directions).float()
    directions = directions * model.fused_neck.weight.detach().cpu()[None, :]
    directions = F.normalize(directions, dim=1) * norms[:, None]
    with torch.no_grad():
        weight.copy_(directions.to(weight.device))
    after = weight.detach().cpu()
    norm_error = float((after.norm(dim=1) - norms).abs().max())
    assert norm_error < 1e-6 and not torch.equal(before, after)
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
    model.identity_head_initialization = dict(
        status='ACTUAL_BUILD_CLASSIFIER_TRAIN_BANK_INITIALIZED', bank=str(BANK_PATH), bank_sha256=BANK_SHA,
        bank_bytes=BANK_BYTES, teacher_sha256=bank['teacher_sha256'], training_records=8675, identities=50,
        parameters_added=0, changed_tensor='fused_classifier.weight', other_head_and_BN_parameters_unchanged=True,
        old_row_norms=norms.tolist(), row_norm_max_error=norm_error, CPU_CUDA_RNG_exact=True,
        old_weight_sha256=hashlib.sha256(before.numpy().tobytes()).hexdigest(),
        initialized_weight_sha256=hashlib.sha256(after.numpy().tobytes()).hexdigest(),
        initialization_encoder_forwards=0, initialization_optimizer_updates=0,
        source='Cached full-official TRAIN unit5120 teacher features; true training IDs. Center then population-variance/BN-epsilon whitening; existing BN gamma; retain each original random weight row norm.',
        limits='Build-time classifier directions only, then normal trainable updates. Full TRAIN bank includes each training sample; not held-out classification or retrieval evidence. Cached eval transforms omit training augment/expert deltas. Batch BN statistics differ from population whitening; no guaranteed native loss or unseen-identity gain. No query/gallery features or labels used.')
    model.anchor_record['identity_classifier_initialization'] = dict(model.identity_head_initialization)
    return model
