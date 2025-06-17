#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

import torchscratch as ts

def test_simple():
    """Test simple operations to isolate the segfault"""
    print("Testing simple operations...")
    
    try:
        print("1. Creating tensor...")
        x = ts.tensor([[1.0, 2.0]], requires_grad=True)
        print(f"   x shape: {x.shape()}")
        
        print("2. Creating linear layer...")
        linear = ts.nn.Linear(2, 1)
        print(f"   Linear created")
        
        print("3. Forward pass...")
        output = linear(x)
        print(f"   Output shape: {output.shape()}")
        
        print("4. Creating target...")
        target = ts.tensor([[1.0]], requires_grad=False)
        print(f"   Target shape: {target.shape()}")
        
        print("5. Computing loss...")
        loss = ts.nn.mse_loss(output, target)
        print(f"   Loss computed: {loss}")
        
        print("6. Creating optimizer...")
        params = linear.parameters()
        print(f"   Found {len(params)} parameters")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple()
