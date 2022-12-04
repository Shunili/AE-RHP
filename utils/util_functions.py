import numpy as np
import torch

def seq_one_hot_AE(seq, monomer_codes, max_length):
    '''Convert a RHP sequence string into one-hot encoded pytorch tensor.'''
    #Initialize 
    tensor = torch.zeros(len(monomer_codes) + 1, max_length + 1)
        
    #Pad With Termination Characters
    tensor[-1, len(seq):] = 1
    
    #Encode Tensor
    for pi, monomer in enumerate(seq):
        monomer_index = monomer_codes.find(monomer)
        tensor[monomer_index][pi] = 1
        
    return tensor

def compute_HLB_sseq_AE(seq, win_len, aadict, max_length):     
    # Method 1:
    # padding = '0' * (max_length - len(seq))
    # seq_padded = seq + padding
    
    # bins = [aadict[str(k)] for k in seq_padded]
    # sliding_array = []
    # for i in range(len(seq_padded) - win_len + 1):
    # sliding_array.append(bins[i:i + win_len]) #n by win_len
    # HLB = torch.tensor(np.mean(sliding_array, axis = 1))
    
    # Method 2:
    bins = [aadict[str(k)] for k in seq]
    sliding_array = []
    for i in range(len(seq) - win_len + 1):
        sliding_array.append(bins[i:i + win_len]) #n by win_len
    HLB = torch.tensor(np.mean(sliding_array, axis = 1))
    
    # target_len = max_length - win_len + 1 - (len(seq) - win_len + 1)
    # padding = [9] * (max_length - len(seq))
    # new = torch.cat((HLB, torch.tensor(padding, dtype=torch.float64)))

    padding = -1 * torch.ones(max_length - len(seq), dtype = torch.float64)
    updated = torch.cat((HLB, padding))
    return updated