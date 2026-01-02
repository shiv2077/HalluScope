"""
Seeding Utilities Module.

This module provides utilities for setting random seeds
to ensure reproducibility across experiments.
"""

from typing import Optional
import random

# Global seed storage
_current_seed: Optional[int] = None


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Sets seed for:
    - Python random
    - NumPy
    - PyTorch (if available)
    
    Args:
        seed: Random seed value.
    """
    # TODO: Implement seed setting
    global _current_seed
    _current_seed = seed
    
    # Python random
    random.seed(seed)
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For full reproducibility (may impact performance)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_seed() -> Optional[int]:
    """
    Get the current global seed.
    
    Returns:
        Current seed or None if not set.
    """
    return _current_seed


def generate_seed() -> int:
    """
    Generate a random seed.
    
    Returns:
        Random seed value.
    """
    return random.randint(0, 2**32 - 1)


class SeedContext:
    """
    Context manager for temporary seed setting.
    
    Useful for reproducible sections within larger code.
    """
    
    def __init__(self, seed: int):
        """
        Initialize seed context.
        
        Args:
            seed: Seed to use within context.
        """
        self.seed = seed
        self.previous_seed = None
    
    def __enter__(self):
        """Enter context and set seed."""
        self.previous_seed = get_seed()
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous seed."""
        if self.previous_seed is not None:
            set_seed(self.previous_seed)
        return False


def make_reproducible(seed: int = 42) -> None:
    """
    Configure environment for maximum reproducibility.
    
    Note: This may impact performance.
    
    Args:
        seed: Random seed.
    """
    # TODO: Implement full reproducibility setup
    set_seed(seed)
    
    # Set environment variables
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch specific
    try:
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
