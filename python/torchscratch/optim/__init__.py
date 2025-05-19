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

# Import Python implementations (these will override any C++ implementations with the same name)
# from .sgd import SGD
# from .adam import Adam

__all__ = [
    # "SGD",
    # "Adam"
]