# Externals
import torch
import torch.nn as nn
from collections import OrderedDict
# from torch.nn.parallel import DistributedDataParallel

# Locals
from .base import BaseTrainer
from models import get_model
import utils.losses

class AE_trainer_MP(BaseTrainer):
    """
    Trainer for AB_seq_LSTM model
    This completes the logic of build, train_epoch evaluate etc.
    """
    def __init__(self, **kwargs):
        super(AE_trainer_MP, self).__init__(**kwargs)

    def build(self, config):
        """Instantiate our model, optimizer, loss function"""
        # Construct the model
        self.model = get_model(**config['model']).to(self.device)

        # Construct the loss function
        self.loss_func = utils.losses.get_loss(**config['loss'])

        # Construct the optimizer
        optimizer_config = config['optimizer']
        Optim = getattr(torch.optim, optimizer_config.pop('name'))
        self.optimizer = Optim(self.model.parameters(), **optimizer_config)
        
        # Construct lr scheduler
        if 'lr_scheduler' in config:
            sched_conf = config['lr_scheduler']
            sched_type = getattr(torch.optim.lr_scheduler, sched_conf.pop('name'))
            self.scheduler = sched_type(self.optimizer, **sched_conf)
        else:
            self.scheduler = None 

        # Print a model summary
        self.logger.info(self.model)
        self.logger.info('Number of parameters: %i', sum(p.numel() for p in self.model.parameters()))
    
    def state_dict(self):
        """Trainer state dict for checkpointing"""
        d = dict(self.model.state_dict(),
            optimizer=self.optimizer.state_dict())
        if self.scheduler is not None:
            d.update(scheduler=self.scheduler.state_dict())
        return d

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])

    def train_epoch(self, data_loader, grad_clip = 1):
        """Train for one epoch"""
        self.model.train()

        # Reset metrics
        total_losses = OrderedDict()

        # Loop over training batches
        for i, (batch_input, hlb_true) in enumerate(data_loader):
            batch_input = batch_input.float().to(self.device)
            _, batch_target = batch_input.max(1)

            hlb_true = hlb_true.float().to(self.device)

            self.optimizer.zero_grad()

            batch_output, hlb_pred = self.model(batch_input)
            batch_losses = self.loss_func(batch_output, batch_target, hlb_pred, hlb_true)
            batch_losses['loss'].backward()

            if grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            self.optimizer.step()
            
            for k, v in batch_losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()

        for k in total_losses.keys():
            total_losses[k] /= len(data_loader)

        # Return summary
        return dict(**total_losses)

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""

        self.model.eval()

        # Reset metrics
        total_losses = OrderedDict()

        for i, (batch_input, hlb_true) in enumerate(data_loader):
            batch_input = batch_input.float().to(self.device)
            _, batch_target = batch_input.max(1)

            hlb_true = hlb_true.float().to(self.device)

            batch_output, hlb_pred = self.model(batch_input)
            batch_losses = self.loss_func(batch_output, batch_target, hlb_pred, hlb_true)
            
            for k, v in batch_losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()

        for k in total_losses.keys():
            total_losses[k] /= len(data_loader)
        # Return summary
        return dict(**total_losses)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def get_trainer(**kwargs):
    return AE_trainer_MP(**kwargs)

def _test():
    t = AE_trainer_MP(output_dir='./')
    t.build_model()