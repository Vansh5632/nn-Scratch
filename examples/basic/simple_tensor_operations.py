#!/usr/bin/env python3
"""
Simple TorchScratch Example: Basic Tensor Operations

This example demonstrates how to create tensors and perform basic operations
using the TorchScratch library.
"""

import numpy as np
import torchscratch as ts

def main():
    print("TorchScratch Basic Tensor Operations Example")
    print("-" * 50)
    
    # Create tensors from NumPy arrays
    print("Creating tensors from NumPy arrays:")
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    
    a = ts.Tensor(a_np)
    b = ts.Tensor(b_np)
    
    print(f"Tensor a shape: {a.shape()}")
    print(f"Tensor b shape: {b.shape()}")
    
    # Basic tensor operations
    print("\nBasic tensor operations:")
    
    # Addition
    c = ts.add(a, b)
    c_np = c.numpy()
    print(f"a + b = \n{c_np}")
    
    # Multiplication
    d = ts.mul(a, b)
    d_np = d.numpy()
    print(f"a * b = \n{d_np}")
    
    # Matrix multiplication
    e = ts.matmul(a, b)
    e_np = e.numpy()
    print(f"a @ b = \n{e_np}")
    
    # Transpose
    f = ts.transpose(a)
    f_np = f.numpy()
    print(f"a.T = \n{f_np}")
    
    # Autograd example
    print("\nAutograd example:")
    
    # Create variables
    x = ts.Variable(a, requires_grad=True)
    y = ts.Variable(b, requires_grad=True)
    
    # Forward pass
    z = ts.add(x, y)  # z = x + y
    print(f"z = x + y shape: {z.data().shape()}")
    
    # Backward pass (compute gradients)
    z.backward()
    
    # Print gradients
    print(f"x.grad = \n{x.grad().numpy()}")  # Should be all ones
    print(f"y.grad = \n{y.grad().numpy()}")  # Should be all ones
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()