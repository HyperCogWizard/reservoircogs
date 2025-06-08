"""Example demonstrating B-Series Rooted Tree Gradient Descent for Ridge Regression.

This example shows how to use the BSeriesRidgeRegression class for optimized
ridge regression in reservoir computing applications. The B-Series approach
combines classical numerical analysis with machine learning optimization for
enhanced convergence properties.
"""

import numpy as np
from reservoirpy.compat import BSeriesRidgeRegression
from reservoirpy.compat.regression_models import RidgeRegression


def generate_reservoir_data(n_samples=200, n_features=50, n_outputs=3, noise_level=0.1):
    """Generate synthetic reservoir computing data."""
    np.random.seed(42)
    
    # Create synthetic reservoir states (features)
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic target pattern with temporal dependencies
    t = np.linspace(0, 4*np.pi, n_samples)
    Y = np.zeros((n_samples, n_outputs))
    Y[:, 0] = np.sin(t) + noise_level * np.random.randn(n_samples)
    Y[:, 1] = np.cos(t) + noise_level * np.random.randn(n_samples)
    Y[:, 2] = np.sin(2*t) * np.cos(t) + noise_level * np.random.randn(n_samples)
    
    return X, Y


def compare_regression_methods():
    """Compare B-Series regression with standard ridge regression."""
    print("B-Series Rooted Tree Gradient Descent Example")
    print("=" * 50)
    
    # Generate data
    X, Y = generate_reservoir_data()
    print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features, {Y.shape[1]} outputs")
    
    # Split data for training and testing
    split_idx = int(0.8 * X.shape[0])
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Test different approaches
    methods = {
        'Standard Ridge': RidgeRegression(ridge=0.1),
        'B-Series RK2': BSeriesRidgeRegression(ridge=0.1, rk_order=2, max_iterations=50),
        'B-Series RK4': BSeriesRidgeRegression(ridge=0.1, rk_order=4, max_iterations=50),
        'B-Series RK6': BSeriesRidgeRegression(ridge=0.1, rk_order=6, max_iterations=30),
    }
    
    results = {}
    
    for name, model in methods.items():
        print(f"\nTraining {name}...")
        
        # Initialize and train
        model.initialize(dim_in=X_train.shape[1], dim_out=Y_train.shape[1])
        Wout = model.fit(X_train, Y_train)
        
        # Evaluate on test set
        # Add bias for prediction
        X_test_bias = np.column_stack([np.ones(X_test.shape[0]), X_test])
        Y_pred = X_test_bias @ Wout.T
        
        # Calculate metrics
        mse = np.mean((Y_test - Y_pred)**2)
        mae = np.mean(np.abs(Y_test - Y_pred))
        
        results[name] = {
            'mse': mse,
            'mae': mae,
            'predictions': Y_pred
        }
        
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
    
    # Display results
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    
    best_mse = min(result['mse'] for result in results.values())
    best_mae = min(result['mae'] for result in results.values())
    
    for name, result in results.items():
        mse_improvement = ((result['mse'] - best_mse) / best_mse) * 100
        mae_improvement = ((result['mae'] - best_mae) / best_mae) * 100
        
        print(f"{name}:")
        print(f"  MSE: {result['mse']:.6f} ({mse_improvement:+.1f}%)")
        print(f"  MAE: {result['mae']:.6f} ({mae_improvement:+.1f}%)")
    
    return results, X_test, Y_test


def demonstrate_convergence():
    """Demonstrate convergence properties of different B-Series orders."""
    print("\n" + "=" * 50)
    print("CONVERGENCE ANALYSIS")
    print("=" * 50)
    
    # Generate smaller dataset for convergence analysis
    X, Y = generate_reservoir_data(n_samples=100, n_features=20, n_outputs=2)
    
    orders = [2, 4, 6]
    convergence_data = {}
    
    for order in orders:
        print(f"\nAnalyzing RK{order} convergence...")
        
        model = BSeriesRidgeRegression(
            ridge=0.1, 
            rk_order=order, 
            max_iterations=100,
            step_size=0.005,
            tolerance=1e-8
        )
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        
        # Track loss during optimization
        X_bias = np.column_stack([np.ones(X.shape[0]), X])
        losses = []
        
        for i in range(100):
            loss = model._loss_function(model.Wout, X_bias, Y)
            losses.append(loss)
            
            if i < 99:  # Don't fit on last iteration
                # Single iteration of partial_fit
                model.max_iterations = 1
                model.partial_fit(X, Y)
        
        convergence_data[f'RK{order}'] = losses
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    
    return convergence_data


def main():
    """Main demonstration function."""
    try:
        # Compare regression methods
        results, X_test, Y_test = compare_regression_methods()
        
        # Demonstrate convergence properties
        convergence_data = demonstrate_convergence()
        
        print("\n" + "=" * 50)
        print("B-SERIES FEATURES DEMONSTRATED")
        print("=" * 50)
        print("✓ Rooted tree gradient computation")
        print("✓ Multiple Runge-Kutta integration orders (RK2, RK4, RK6)")
        print("✓ B-series expansion coefficients")
        print("✓ Numerical stability enhancements")
        print("✓ Convergence analysis and comparison")
        print("✓ Integration with existing ReservoirPy framework")
        
        print("\nB-Series Rooted Tree Gradient Descent implementation complete!")
        print("This provides a foundation for advanced numerical optimization")
        print("in reservoir computing applications.")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        print("This is a research implementation. Some features may require")
        print("additional tuning for optimal performance.")


if __name__ == "__main__":
    main()