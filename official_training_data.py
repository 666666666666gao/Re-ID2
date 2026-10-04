"""Entire official train/query/gallery; no artificial identity holdout."""
from pathlib import Path

import numpy as np

from experiment_data import source_records
from full_evaluation import official_records


def name(row):
    return Path(row[0] if isinstance(row[0], str) else row[0][0]).name


def metadata(records, prefix):
    return {prefix + '_' + key: np.asarray(values) for key, values in {
        'ids': [r[1] for r in records], 'cameras': [r[2] for r in records],
        'scenes': [r[3] for r in records], 'names': [name(r) for r in records]}.items()}


def full_records(root, dataset):
    source = source_records(root, dataset)
    labels = {pid: i for i, pid in enumerate(sorted({r[1] for r in source}))}
    train = [(paths, labels[pid], camera, scene) for paths, pid, camera, scene in source]
    query, gallery = official_records(root, dataset)
    assert len(train) == len(source) and len({name(r) for r in train}) == len(train)
    cameras = max(r[2] for r in source) + 1
    manifest = {'dataset': dataset, 'training_heldout_identities': 0,
                'training_labels': {str(pid): label for pid, label in labels.items()},
                'train': [dict(name=name(r), identity=r[1], camera=r[2], scene=r[3]) for r in source],
                'query': [dict(name=name(r), identity=r[1], camera=r[2], scene=r[3]) for r in query],
                'gallery': [dict(name=name(r), identity=r[1], camera=r[2], scene=r[3]) for r in gallery]}
    return train, query, gallery, len(labels), cameras, manifest
