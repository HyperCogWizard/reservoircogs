#!/usr/bin/env python3
"""
Example: J-Surface Julia Differential Equations for Reservoir Computing

This example demonstrates the use of Julia-based differential equation solving
with elementary differential echo state networks for time series prediction.

The implementation automatically falls back to Python when Julia is not available.
"""

import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.compat import JuliaDifferentialESN


def generate_mackey_glass(n_timesteps=1000, tau=17, beta=0.2, gamma=0.1, n=10, dt=0.1):
    """Generate Mackey-Glass time series for testing."""
    x = np.zeros(n_timesteps)
    x[0] = 1.2  # Initial condition
    
    # Simple discrete approximation
    for i in range(1, n_timesteps):
        if i >= tau:
            dx = beta * x[i - tau] / (1 + x[i - tau]**n) - gamma * x[i - 1]
        else:
            dx = beta * 1.2 / (1 + 1.2**n) - gamma * x[i - 1]
        x[i] = x[i - 1] + dt * dx
    
    return x


def create_input_output_sequences(data, input_length=3, prediction_horizon=1):
    """Create input-output sequences for prediction."""
    X, Y = [], []
    
    for i in range(len(data) - input_length - prediction_horizon + 1):
        X.append(data[i:i + input_length])
        Y.append(data[i + input_length:i + input_length + prediction_horizon])
    
    return np.array(X), np.array(Y)


def main():
    print("J-Surface Julia Differential Equations Example")
    print("=" * 50)
    
    # Generate test data (Mackey-Glass chaotic time series)
    print("Generating Mackey-Glass time series...")
    timeseries = generate_mackey_glass(n_timesteps=800)
    
    # Prepare input-output sequences
    print("Preparing input-output sequences...")
    X, Y = create_input_output_sequences(timeseries, input_length=3, prediction_horizon=1)
    
    # Split into train and test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    print(f"Training data: {X_train.shape[0]} sequences")
    print(f"Test data: {X_test.shape[0]} sequences")
    
    # Create and configure the Julia Differential ESN
    print("\nCreating Julia Differential ESN...")
    model = JuliaDifferentialESN(
        n_reservoir=100,
        spectral_radius=0.95,
        input_scaling=1.0,
        leaking_rate=0.3,
        ridge=1e-6,
        solver="Tsit5",  # Julia ODE solver
        dt=0.01,         # Integration time step
        seed=42
    )
    
    # Initialize the model
    model.initialize(dim_in=X_train.shape[1], dim_out=Y_train.shape[1])
    
    print(f"Julia available: {model.julia_available}")
    print(f"Reservoir size: {model.n_reservoir}")
    print(f"Spectral radius: {model.spectral_radius}")
    print(f"ODE solver: {model.solver}")
    
    # Train the model
    print("\nTraining the model...")
    model.fit(X_train, Y_train)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Calculate error metrics
    mse = np.mean((predictions - Y_test)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - Y_test))
    
    print(f"\nPerformance Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    # Plot results
    print("\nGenerating plots...")
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Original time series
    plt.subplot(2, 2, 1)
    plt.plot(timeseries[:200])
    plt.title("Mackey-Glass Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    # Plot 2: Training data sample
    plt.subplot(2, 2, 2)
    plt.plot(range(len(X_train)), Y_train.flatten()[:len(X_train)], 'b-', label='Training targets')
    plt.title("Training Data Sample")
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.legend()
    
    # Plot 3: Predictions vs actual
    plt.subplot(2, 2, 3)
    test_length = min(100, len(Y_test))
    plt.plot(range(test_length), Y_test.flatten()[:test_length], 'g-', label='Actual', linewidth=2)
    plt.plot(range(test_length), predictions.flatten()[:test_length], 'r--', label='Predicted', linewidth=2)
    plt.title("Predictions vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    
    # Plot 4: Error over time
    plt.subplot(2, 2, 4)
    errors = np.abs(predictions.flatten() - Y_test.flatten())[:test_length]
    plt.plot(range(test_length), errors, 'orange', linewidth=1)
    plt.title("Prediction Error")
    plt.xlabel("Time")
    plt.ylabel("Absolute Error")
    
    plt.tight_layout()
    plt.savefig('julia_de_example_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'julia_de_example_results.png'")
    
    # Test different configurations
    print("\nTesting different configurations...")
    
    configs = [
        {"solver": "Tsit5", "dt": 0.01, "leaking_rate": 0.3},
        {"solver": "Vern7", "dt": 0.005, "leaking_rate": 0.5},
        {"solver": "Rodas5P", "dt": 0.02, "leaking_rate": 0.1},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        test_model = JuliaDifferentialESN(
            n_reservoir=50, 
            spectral_radius=0.9,
            seed=42,
            **config
        )
        test_model.initialize(dim_in=X_train.shape[1], dim_out=Y_train.shape[1])
        test_model.fit(X_train[:100], Y_train[:100])  # Smaller training set for speed
        test_predictions = test_model.predict(X_test[:50])
        test_rmse = np.sqrt(np.mean((test_predictions - Y_test[:50])**2))
        print(f"RMSE: {test_rmse:.6f}")
    
    print("\nExample completed successfully!")
    print("\nNext steps:")
    print("- Try different reservoir sizes")
    print("- Experiment with different ODE solvers") 
    print("- Test on your own time series data")
    print("- Install Julia + DifferentialEquations.jl for improved performance")


if __name__ == "__main__":
    main()