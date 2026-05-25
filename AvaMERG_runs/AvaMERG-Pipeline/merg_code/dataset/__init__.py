from header import *
from .samplers import DistributedBatchSampler
import torch
import importlib
from dataset.all_dataset import multimodal_empathetic_dialogue


def load_dataset(args):
    _dataset = multimodal_empathetic_dialogue(args)

    world_size = torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1 if torch.distributed.is_available() and torch.distributed.is_initialized() else 1
    rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0 if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
    if args['mode'] == 'train':
        batch_size = args.get('world_size', 1) * (args.get('dschf').config['train_micro_batch_size_per_gpu'] if args.get('dschf', None) is not None else 2)
    elif args['mode'] == 'test':
        batch_size = 1
    else:
        raise ValueError(" Mode Error! The mode should be train or test! ")
    sampler = torch.utils.data.RandomSampler(_dataset)
    batch_sampler = DistributedBatchSampler(
                                                sampler=sampler,
                                                batch_size=batch_size,
                                                drop_last=True,
                                                rank=rank,
                                                world_size=world_size,
                                            )
    iter_ = DataLoader(
        _dataset,
        batch_sampler=batch_sampler,
        num_workers=1,
        collate_fn=_dataset.collate_fn,
        pin_memory=True
    )
    return _dataset, iter_, sampler

