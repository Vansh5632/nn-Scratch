"""
TorchScratch: A PyTorch-like library built from scratch for educational purposes.

This library helps you understand the inner workings of deep learning frameworks
by implementing tensor operations, automatic differentiation, and neural networks
from the ground up.
"""

__version__ = "0.1.0"

# Import C++ extension module
try:
    import torchscratch_cpp
except ImportError as e:
    raise ImportError(
        "Failed to import C++ extension module 'torchscratch_cpp'. "
        "Make sure the library is properly built and installed. "
        "See setup instructions in README.md for more information."
    ) from e

# Import key classes and functions for easy access
from torchscratch_cpp import (
    Tensor,
    Variable,
    add,
    mul,
    matmul,
    transpose,
)

# Import submodules
from . import nn
from . import optim

__all__ = [
    "Tensor",
    "Variable",
    "add",
    "mul",
    "matmul",
    "transpose",
    "nn",
    "optim",
]