#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

import torchscratch as ts

def test_basic_neural_network():
    """Test basic neural network components"""
    print("Testing Neural Network Components")
    print("=" * 40)
    
    # Test tensor creation
    print("\n1. Testing tensor creation...")
    x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = ts.tensor([[0.5], [1.5]], requires_grad=False)
    print(f"Input tensor x: shape {x.shape()}")
    print(f"Target tensor y: shape {y.shape()}")
    
    # Test linear layer
    print("\n2. Testing linear layer...")
    linear = ts.nn.Linear(2, 1)
    print(f"Linear layer created: {linear}")
    print(f"Weight shape: {linear.weight().shape()}")
    print(f"Bias shape: {linear.bias().shape()}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    output = linear(x)
    print(f"Linear output shape: {output.shape()}")
    
    # Test activation functions
    print("\n4. Testing activation functions...")
    relu_out = ts.nn.relu(output)
    print(f"ReLU output shape: {relu_out.shape()}")
    
    sigmoid_out = ts.nn.sigmoid(output)
    print(f"Sigmoid output shape: {sigmoid_out.shape()}")
    
    # Test loss function
    print("\n5. Testing loss function...")
    loss = ts.nn.mse_loss(sigmoid_out, y)
    print(f"MSE loss: {loss}")
    
    # Test optimizer
    print("\n6. Testing optimizer...")
    params = linear.parameters()
    optimizer = ts.optim.SGD(params, lr=0.01)
    print(f"SGD optimizer created with {len(params)} parameters")
    
    # Test backward pass
    print("\n7. Testing backward pass...")
    optimizer.zero_grad()
    loss.backward()
    print("Backward pass completed")
    
    # Test parameter update
    print("\n8. Testing parameter update...")
    optimizer.step()
    print("Parameter update completed")
    
    print("\nAll neural network components working correctly!")

if __name__ == "__main__":
    test_basic_neural_network()
