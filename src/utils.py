import torch.nn as nn
import torch
import shutil
from torch.utils.data import Dataset
import numpy as np

class RandomTensorDataset(Dataset):
    def __init__(self, num_samples, N, C):
        """
        Args:
            num_samples (int): number of sample.
            N (int): number of tokens.
            C (int): token dimension.
        """
        self.num_samples = num_samples
        self.N = N
        self.C = C
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = torch.randn(self.N, self.C)
        return sample
    
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def save_checkpoint(state, is_best, save_dir, filename="checkpoint.pth.tar"):
    torch.save(state, f'{save_dir}/{filename}')
    if is_best:
        shutil.copyfile(f'{save_dir}/{filename}', f'{save_dir}/checkpoint_best_loss.pth.tar')


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)