"""
Optimization algorithms for TorchScratch.

This module provides optimizers for training neural networks.
"""

# Import from C++ extension
try:
    from torchscratch_cpp.optim import *
except ImportError:
    # If optim components aren't yet implemented in C++, we'll use Python implementations
    pass

__all__ = [
    "SGD"
]