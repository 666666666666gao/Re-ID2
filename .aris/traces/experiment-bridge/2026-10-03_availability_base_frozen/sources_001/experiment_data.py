"""Fixed identity-disjoint development splits using installed training data."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from data.datasets.RGBNT201 import RGBNT201
from data.datasets.RGBNT100 import RGBNT100
from data.datasets.msvr310 import MSVR310
from data.datasets.sampler import RandomIdentitySampler
from data.datasets.make_dataloader import RandomErasing


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_records(root, name):
    subdir, cls = {
        'RGBNT201': ('RGBNT201/train_171', RGBNT201),
        'RGBNT100': ('RGBNT100/rgbir/bounding_box_train', RGBNT100),
        'MSVR310': ('MSVR310/bounding_box_train', MSVR310),
    }[name]
    return sorted(cls._process_dir(None, str(Path(root) / subdir), relabel=False), key=lambda r: str(r[0]))


def split_records(root, name):
    records = source_records(root, name)
    split = json.loads(Path('splits.json').read_text())
    dev_ids = set(split[name]['dev_ids'])
    train = [r for r in records if r[1] not in dev_ids]
    dev = [r for r in records if r[1] in dev_ids]
    labels = {pid: i for i, pid in enumerate(sorted({r[1] for r in train}))}
    fit = [(p, labels[pid], cam, scene) for p, pid, cam, scene in train]
    selector = 3 if name == 'MSVR310' else 2
    query = [i for i, row in enumerate(dev) if any(other[1] == row[1] and other[selector] != row[selector] for other in dev)]
    assert dev and query and set(labels).isdisjoint(dev_ids)
    cameras = max(r[2] for r in records) + 1
    return fit, dev, query, len(labels), cameras


class Triplets(Dataset):
    def __init__(self, records, cfg, training):
        self.records = records
        self.training = training
        interpolation = T.InterpolationMode.BICUBIC if training else T.InterpolationMode.BILINEAR
        operations = [T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=interpolation)]
        if training:
            operations += [T.RandomHorizontalFlip(cfg.INPUT.PROB), T.Pad(cfg.INPUT.PADDING), T.RandomCrop(cfg.INPUT.SIZE_TRAIN)]
        operations += [T.ToTensor(), T.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)]
        if training:
            operations += [RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', device='cpu')]
        self.transform = T.Compose(operations)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        paths, pid, cam, scene = self.records[index]
        if isinstance(paths, str):
            with Image.open(paths) as raw:
                montage = raw.convert('RGB')
            assert montage.size == (768, 128)
            images = [montage.crop((m * 256, 0, (m + 1) * 256, 128)) for m in range(3)]
            name = Path(paths).name
        else:
            images = []
            for path in paths:
                with Image.open(path) as raw:
                    images.append(raw.convert('RGB'))
            name = Path(paths[0]).name
        # Use the same spatial crop, flip and erasing draw for the paired modalities.
        rng_python, rng_torch = random.getstate(), torch.get_rng_state()
        result = []
        for image in images:
            random.setstate(rng_python)
            torch.set_rng_state(rng_torch)
            result.append(self.transform(image))
        return dict(zip(('RGB', 'NI', 'TI'), result)), pid, cam, scene, name


def make_loader(records, cfg, training, seed):
    dataset = Triplets(records, cfg, training)
    sampler = RandomIdentitySampler(records, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE) if training else None
    return DataLoader(dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH if training else 64,
                      sampler=sampler, shuffle=False, num_workers=4, drop_last=training,
                      generator=torch.Generator().manual_seed(seed), pin_memory=True)
