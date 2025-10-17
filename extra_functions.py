import numpy as np
import random
import torch 

def set_seed(seed: int = 666):

    """
    Sets the random seed across Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int, optional): The seed value to use. Defaults to 666.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_metrics(metrics_dict):

    """
    Returns new dictionary with the same keys and Tensor metric values converted to Python floats.

    Args:
        metrics_dict (dict): Dictionary where keys are metric names and values are lists 
            containing numbers or PyTorch tensors.

    Example:
        Input: {'loss': [tensor(0.5), tensor(0.3)], 'acc': [0.8, tensor(0.9)]}
        Output: {'loss': [0.5, 0.3], 'acc': [0.8, 0.9]}
    """

    cleaned = {}

    for key, values in metrics_dict.items():
        cleaned_values = []
        for v in values:
            if isinstance(v, torch.Tensor):
                cleaned_values.append(v.item())
            else:
                cleaned_values.append(float(v))
        cleaned[key] = cleaned_values

    return cleaned