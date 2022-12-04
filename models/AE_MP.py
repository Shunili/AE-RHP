import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hiddens=[]):
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        if isinstance(output_shape, int):
            output_shape = (output_shape,)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hiddens = hiddens

        model = []
        prev_h = np.prod(input_shape)
        for h in hiddens + [np.prod(output_shape)]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.pop()
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x)
    
    
class autoencoder(nn.Module):
    def __init__(self, input_dim = [5, 51], latent_dim = 16, win_length = 5,
                 enc_hidden_sizes=[512, 256, 64], dec_hidden_sizes=[64, 256, 512], 
                 compute_HLB = True, MLP_HLB_size = [32, 32]):
        super(autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.win_length = win_length
        self.compute_HLB = compute_HLB

        self.encoder = MLP(input_dim, latent_dim, enc_hidden_sizes)
        self.decoder = MLP(latent_dim, input_dim, dec_hidden_sizes) 
        
        if self.compute_HLB:
            self.MLP_HLB = MLP(latent_dim, input_dim[1] - win_length, MLP_HLB_size)
    
    def forward(self, x):

        N, C, L = x.shape
        x_flat = x.flatten(start_dim=1)
 
        latents = self.encoder(x_flat)
        out = self.decoder(latents).reshape(N, C, -1)

        if self.compute_HLB:
            hlb_pred = self.MLP_HLB(latents)
        else:
            hlb_pred = None
        
        return out, hlb_pred


def build_model(**kwargs):
    return autoencoder(**kwargs)