#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

import torchscratch as ts

def test_complete_training():
    """Test a complete neural network training example"""
    print("Testing Complete Neural Network Training")
    print("=" * 50)
    
    # Create a simple dataset (XOR problem)
    # Input: 2D points, Output: XOR of the coordinates (0 or 1)
    X = ts.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], requires_grad=False)
    y = ts.tensor([[0.0], [1.0], [1.0], [0.0]], requires_grad=False)
    
    print(f"Dataset:")
    print(f"  Input shape: {X.shape()}")
    print(f"  Target shape: {y.shape()}")
    
    # Create a 2-layer neural network: 2 -> 4 -> 1
    print(f"\nNeural Network Architecture:")
    layer1 = ts.nn.Linear(2, 4)  # 2 inputs -> 4 hidden
    layer2 = ts.nn.Linear(4, 1)  # 4 hidden -> 1 output
    print(f"  Layer 1: {layer1.in_features()} -> {layer1.out_features()}")
    print(f"  Layer 2: {layer2.in_features()} -> {layer2.out_features()}")
    
    # Create optimizer with all parameters
    parameters = layer1.parameters() + layer2.parameters()
    optimizer = ts.optim.SGD(parameters, lr=0.5)  # Higher learning rate for faster convergence
    print(f"  Optimizer: SGD with {len(parameters)} parameters, lr=0.5")
    
    # Training loop
    print(f"\nTraining Loop:")
    num_epochs = 200
    print_every = 50
    
    for epoch in range(num_epochs):
        # Forward pass
        h1 = layer1(X)          # [4, 2] @ [2, 4]^T -> [4, 4]
        h1_relu = ts.nn.relu(h1)    # ReLU activation
        h2 = layer2(h1_relu)        # [4, 4] @ [4, 1]^T -> [4, 1]  
        predictions = ts.nn.sigmoid(h2)  # Sigmoid activation
        
        # Compute loss
        loss = ts.nn.binary_cross_entropy_loss(predictions, y)
        
        # Print progress
        if epoch % print_every == 0:
            loss_val = loss.item()
            print(f"  Epoch {epoch:3d}: Loss = {loss_val:.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update parameters
        optimizer.step()
    
    # Final evaluation
    print(f"\nFinal Evaluation:")
    with ts.no_grad():
        h1 = layer1(X)
        h1_relu = ts.nn.relu(h1)
        h2 = layer2(h1_relu)
        final_predictions = ts.nn.sigmoid(h2)
        final_loss = ts.nn.binary_cross_entropy_loss(final_predictions, y)
        
        print(f"  Final Loss: {final_loss.item():.6f}")
        print(f"  Predictions vs Targets:")
        
        # Extract predictions (need to access the raw data)
        pred_data = final_predictions.data()
        target_data = y.data()
        input_data = X.data()
        
        for i in range(4):
            # Get input values
            x1 = input_data.data_ptr()[i * 2]
            x2 = input_data.data_ptr()[i * 2 + 1]
            
            # Get prediction and target
            pred = pred_data.data_ptr()[i]
            target = target_data.data_ptr()[i]
            
            # Convert prediction to binary (threshold at 0.5)
            pred_binary = 1 if pred > 0.5 else 0
            
            print(f"    [{x1:.0f}, {x2:.0f}] -> Pred: {pred:.4f} ({pred_binary}) | Target: {target:.0f}")
    
    print(f"\nTraining completed successfully!")
    print(f"The network should learn to approximate the XOR function:")
    print(f"  [0,0] -> 0, [0,1] -> 1, [1,0] -> 1, [1,1] -> 0")

if __name__ == "__main__":
    test_complete_training()
