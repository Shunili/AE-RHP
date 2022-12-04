"""
PyTorch dataset specifications.
"""
import os
import importlib
from torch.utils.data import DataLoader


def get_datasets(name, **data_args):
    """Factory function for importing datasets from local modules"""
    module = importlib.import_module('.' + name, 'datasets')
    return module.get_datasets(**data_args)

def get_collate_fn(name, **data_args):
    """Factory function for importing datasets from local modules"""
    module = importlib.import_module('.' + name, 'datasets')
    return module.get_collate_fn(**data_args)


def get_data_loaders(name, batch_size, collate, **dataset_args):
    """Construct training and validation datasets and data loaders"""

    # Get the datasets
    train_dataset, valid_dataset = get_datasets(name=name, **dataset_args)

    # Get collate fn for dynamic padding
    if collate:
        collate_fn = get_collate_fn(name=name, **dataset_args)
    else:
        collate_fn = None
   
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_loader, valid_loader