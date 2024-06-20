import torch
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def set_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_precision(precision):
    torch.set_default_dtype(precision)

def create_directories():
    for path in ['models', 'errors', 'plots']:
        if not os.path.exists(path):
           os.makedirs(path)