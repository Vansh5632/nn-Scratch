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
    sub,
    mul,
    matmul,
    transpose,
    tensor,
)

# Import submodules
from . import nn
from . import optim

# Add no_grad context manager
class no_grad:
    """Context manager for disabling gradient computation"""
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

__all__ = [
    "Tensor",
    "Variable",
    "add",
    "sub",
    "mul",
    "matmul",
    "transpose",
    "tensor",
    "no_grad",
    "nn",
    "optim",
]