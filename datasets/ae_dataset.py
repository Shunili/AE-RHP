import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import pandas as pd
import os
from utils.util_functions import seq_one_hot_AE, compute_HLB_sseq_AE

class AEDataset(Dataset):
    def __init__(self, seqs, max_length, monomer_codes, hlbdict, win_length, compute_HLB, transform=None):
        
        # set size
        self.size = len(seqs)
        
        # create seq tensors
        one_hot = lambda x: seq_one_hot_AE(x, monomer_codes = monomer_codes, max_length=max_length)
        self.seqs_tensor = torch.stack([one_hot(seq) for seq in seqs])
        
        # create HLB tensors
        if compute_HLB:
            hlb = lambda x: compute_HLB_sseq_AE(x, win_len = win_length, aadict = hlbdict, max_length = max_length)
            self.HLB_values_tensor = torch.stack([hlb(seq) for seq in seqs])
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        seq = self.seqs_tensor[idx]
        hlb_value = self.HLB_values_tensor[idx]
        return seq, hlb_value
    
def get_datasets(data_path, train_val_split, seed, max_length, monomer_codes, win_length, hlbdict,compute_HLB, **kwargs):
    
    data_path = os.path.expandvars(data_path)
    data = pd.read_csv(data_path)

    # Deterministically shuffle the dataset
    all_seqs = data['sequence'].sample(frac=1., random_state=seed)

    # Split into train, val
    seq_train = all_seqs.iloc[:int(len(data) * train_val_split)]
    seq_valid = all_seqs.iloc[int(len(data) * train_val_split):]

    # Create the PyTorch datasets
    train_dataset = AEDataset(seqs = seq_train, max_length = max_length, monomer_codes = monomer_codes,
                              hlbdict=hlbdict, win_length = win_length, compute_HLB=compute_HLB)
    valid_dataset = AEDataset(seqs = seq_valid, max_length = max_length, monomer_codes = monomer_codes,
                               hlbdict=hlbdict, win_length = win_length, compute_HLB=compute_HLB)
    
    return train_dataset, valid_dataset