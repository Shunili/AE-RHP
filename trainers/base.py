# System
import os
import time
import logging

# Externals
import numpy as np
import pandas as pd
import torch

# Locals
import utils

def _format_summary(summary):
    """Make a formatted string for logging summary info"""
    return ' '.join(f'{k} {v:.4g}' for (k, v) in summary.items())

class BaseTrainer(object):
    """
    Base class for PyTorch trainers.
    This implements the common training logic,
    logging of summaries, and checkpoints.
    """

    def __init__(self, output_dir=None, gpu=None, use_wandb=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = (os.path.expandvars(output_dir)
                           if output_dir is not None else None)
        self.gpu = gpu
        if gpu is not None:
            self.device = 'cuda:%i' % gpu
            torch.cuda.set_device(gpu)
        else:
            self.device = 'cpu'
        self.summaries = None
        self.use_wandb = use_wandb

    def _get_summary_file(self):
        return os.path.join(self.output_dir, 'summaries_%i.csv' % 0)

    def save_summary(self, summary, write_file=True):
        """Save new summary information"""

        # First summary
        if self.summaries is None:
            self.summaries = pd.DataFrame([summary])

        # Append a new summary row
        else:
            self.summaries = self.summaries.append([summary], ignore_index=True)

        # Write current summaries to file (note: overwrites each time)
        if write_file and self.output_dir is not None:
            self.summaries.to_csv(self._get_summary_file(), index=False,
                                  float_format='%.6f', sep='\t')

    def load_summaries(self):
        self.summaries = pd.read_csv(self._get_summary_file(), delim_whitespace=True)

    def _get_checkpoint_file(self, checkpoint_id):
        return os.path.join(self.output_dir, 'checkpoints',
                            'checkpoint_%03i.pkl' % checkpoint_id)

    def write_checkpoint(self, checkpoint_id):
        """Write a checkpoint for the trainer"""
        assert self.output_dir is not None
        checkpoint_file = self._get_checkpoint_file(checkpoint_id)
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_id=-1):
        """Load from checkpoint"""
        assert self.output_dir is not None

        # First, load the summaries
        try:
            self.load_summaries()
        except FileNotFoundError:
            self.logger.info('No summaries file found. Will not load checkpoint')
            return

        # Now load the checkpoint
        if checkpoint_id == -1:
            checkpoint_id = self.summaries.epoch.iloc[-1]
        checkpoint_file = self._get_checkpoint_file(checkpoint_id)
        self.logger.info('Loading checkpoint at %s', checkpoint_file)
        self.load_state_dict(torch.load(checkpoint_file, map_location=self.device))

    def state_dict(self):
        """Virtual method to return state dict for checkpointing"""
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        """Virtual method to load a state dict from a checkpoint"""
        raise NotImplementedError

    def build(self, config):
        """Virtual method to build model, optimizer, etc."""
        raise NotImplementedError

    def train_epoch(self, data_loader):
        """Virtual method to train a model"""
        raise NotImplementedError

    def evaluate(self, data_loader):
        """Virtual method to evaluate a model"""
        raise NotImplementedError

    def train(self, train_data_loader, n_epochs, grad_clip = None, valid_data_loader=None):
        """Run the model training"""

        # Determine starting epoch
        start_epoch = 0 if self.summaries is None else self.summaries.epoch.max() + 1

        # Loop over epochs
        for i in range(start_epoch, n_epochs):

            self.logger.info('Epoch %i', i)
            summary = dict(epoch=i)

            # Train on this epoch
            start_time = time.time()
            train_summary = self.train_epoch(train_data_loader, grad_clip = grad_clip)
            train_summary['time'] = time.time() - start_time
            self.logger.info('Train: %s', _format_summary(train_summary))
            for (k, v) in train_summary.items():
                summary[f'train_{k}'] = v

            # Evaluate on this epoch
            if valid_data_loader is not None:
                start_time = time.time()
                valid_summary = self.evaluate(valid_data_loader)
                valid_summary['time'] = time.time() - start_time
                self.logger.info('Valid: %s', _format_summary(valid_summary))
                for (k, v) in valid_summary.items():
                    summary[f'valid_{k}'] = v

            # Save summary, checkpoint
            self.save_summary(summary)
            if self.output_dir is not None:
                #if (i+1) % 2 == 0:
                self.write_checkpoint(checkpoint_id=i)


        return self.summaries