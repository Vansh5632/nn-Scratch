"""
Neural network module for TorchScratch.

This module provides building blocks for creating neural networks.
"""

# Import from C++ extension
try:
    from torchscratch_cpp.nn import *
except ImportError:
    # If nn components aren't yet implemented in C++, we'll use Python implementations
    pass

__all__ = [
    "Linear",
    "relu", "sigmoid", "tanh",
    "mse_loss", "binary_cross_entropy_loss", "cross_entropy_loss"
]