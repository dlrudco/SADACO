import numpy as np
from typing import Dict, List, Union

def seed_everything(seed: int) -> None:
    r"""Seed everything for reproducibility.
    
    Attributes:
        `os`
        `random`
        `numpy`
        `torch`
    """
    import os
    import random

    import numpy as np
    import torch

    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        # torch.autograd.set_detect_anomaly(False)
        # torch.autograd.profiler.profile(False)
        # torch.autograd.profiler.emit_nvtx(False)
        print(f"Seed everything with {seed}.")
        
        
def min_max_scale(samples, min, max):
    samples = samples-samples.min()
    samples = samples / samples.max()
    samples = (max - min) * samples + min
    return samples
