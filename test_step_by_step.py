#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

import torchscratch as ts

def test_training_step_by_step():
    """Test training step by step to isolate issues"""
    print("Testing Training Step-by-Step")
    print("=" * 40)
    
    try:
        # Create simple data
        print("1. Creating data...")
        X = ts.tensor([[1.0, 2.0]], requires_grad=False)
        y = ts.tensor([[1.0]], requires_grad=False)
        print(f"   Data created: X {X.shape()}, y {y.shape()}")
        
        # Create network
        print("2. Creating network...")
        layer1 = ts.nn.Linear(2, 3)
        layer2 = ts.nn.Linear(3, 1)
        print(f"   Network created: 2->3->1")
        
        # Create optimizer
        print("3. Creating optimizer...")
        params = layer1.parameters() + layer2.parameters()
        optimizer = ts.optim.SGD(params, lr=0.01)
        print(f"   Optimizer created with {len(params)} parameters")
        
        # Single forward pass
        print("4. Forward pass...")
        print(f"   X shape: {X.shape()}")
        print(f"   layer1 weight shape: {layer1.weight().shape()}")
        print(f"   layer1 bias shape: {layer1.bias().shape()}")
        try:
            h1 = layer1(X)
        except Exception as e:
            print(f"Error in layer1 forward: {e}")
            # Let's debug step by step
            weight_t = ts.transpose(layer1.weight().data())
            print(f"   weight_t shape: {weight_t.shape()}")
            matmul_result = ts.matmul(X.data(), weight_t)
            print(f"   matmul result shape: {matmul_result.shape()}")
            print(f"   bias shape: {layer1.bias().data().shape()}")
            raise e
        print(f"   h1 shape: {h1.shape()}")
        h1_relu = ts.nn.relu(h1)
        print(f"   h1_relu shape: {h1_relu.shape()}")
        h2 = layer2(h1_relu)
        print(f"   h2 shape: {h2.shape()}")
        
        # Compute loss
        print("5. Computing loss...")
        loss = ts.nn.mse_loss(h2, y)
        print(f"   Loss: {loss.item():.6f}")
        
        # Backward pass
        print("6. Backward pass...")
        optimizer.zero_grad()
        print("   Gradients zeroed")
        loss.backward()
        print("   Backward completed")
        
        # Parameter update
        print("7. Parameter update...")
        optimizer.step()
        print("   Parameters updated")
        
        print("8. Second iteration...")
        # Try another iteration
        h1 = layer1(X)
        h1_relu = ts.nn.relu(h1)
        h2 = layer2(h1_relu)
        loss2 = ts.nn.mse_loss(h2, y)
        print(f"   Loss after update: {loss2.item():.6f}")
        
        print("\nStep-by-step training test successful!")
        
    except Exception as e:
        print(f"Error at step: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_step_by_step()
