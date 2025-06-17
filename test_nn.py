#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

import torchscratch as ts
import numpy as np

def test_neural_network():
    """Test a simple neural network training example"""
    print("Testing Neural Network Training Pipeline")
    print("=" * 50)
    
    # Create a simple dataset (XOR problem)
    # Input: 2D points, Output: XOR of the coordinates (0 or 1)
    X = ts.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], requires_grad=False)
    y = ts.tensor([[0.0], [1.0], [1.0], [0.0]], requires_grad=False)
    
    print(f"Input shape: {X.shape()}")
    print(f"Target shape: {y.shape()}")
    print(f"Input data: {X}")
    print(f"Target data: {y}")
    
    # Create a simple 2-layer neural network
    # Architecture: 2 -> 4 -> 1
    print("\nCreating neural network...")
    
    # Layer 1: 2 input features -> 4 hidden units
    layer1 = ts.nn.Linear(2, 4)
    print(f"Layer 1 weights shape: {layer1.weight.shape()}")
    print(f"Layer 1 bias shape: {layer1.bias.shape()}")
    
    # Layer 2: 4 hidden units -> 1 output
    layer2 = ts.nn.Linear(4, 1)
    print(f"Layer 2 weights shape: {layer2.weight.shape()}")
    print(f"Layer 2 bias shape: {layer2.bias.shape()}")
    
    # Create optimizer
    parameters = layer1.parameters() + layer2.parameters()
    optimizer = ts.optim.SGD(parameters, lr=0.1)
    print(f"Optimizer created with {len(parameters)} parameters")
    
    # Training loop
    print("\nStarting training...")
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # Forward pass
        h1 = layer1(X)  # Linear transformation
        h1_relu = ts.nn.relu(h1)  # ReLU activation
        output = layer2(h1_relu)  # Final linear layer
        predictions = ts.nn.sigmoid(output)  # Sigmoid activation
        
        # Compute loss
        loss = ts.nn.binary_cross_entropy_loss(predictions, y)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.data().item():.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update parameters
        optimizer.step()
    
    # Final evaluation
    print("\nFinal evaluation:")
    with ts.no_grad():
        h1 = layer1(X)
        h1_relu = ts.nn.relu(h1)
        output = layer2(h1_relu)
        final_predictions = ts.nn.sigmoid(output)
        final_loss = ts.nn.binary_cross_entropy_loss(final_predictions, y)
        
        print(f"Final Loss: {final_loss.data().item():.6f}")
        print("Predictions vs Targets:")
        pred_data = final_predictions.data()
        target_data = y.data()
        
        for i in range(4):
            pred_val = pred_data.data_ptr()[i] if hasattr(pred_data, 'data_ptr') else pred_data[i]
            target_val = target_data.data_ptr()[i] if hasattr(target_data, 'data_ptr') else target_data[i]
            print(f"  Input: {X.data().data_ptr()[i*2]:.1f}, {X.data().data_ptr()[i*2+1]:.1f} -> "
                  f"Pred: {pred_val:.4f}, Target: {target_val:.1f}")
    
    print("\nNeural network training completed successfully!")

if __name__ == "__main__":
    test_neural_network()
