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

# Import Python implementations (these will override any C++ implementations with the same name)
# from .linear import Linear
# from .activation import ReLU, Sigmoid, Tanh
# from .loss import MSELoss, CrossEntropyLoss

__all__ = [
    # "Linear",
    # "ReLU", "Sigmoid", "Tanh",
    # "MSELoss", "CrossEntropyLoss"
]