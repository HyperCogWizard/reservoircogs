"""
BPJ Integration Example: Agent-Arena-Relation Pattern with OEIS A000081

This example demonstrates the integration of B-Series, P-Systems, and J-Surfaces
using the Agent-Arena-Relation (AAR) pattern with OEIS A000081 enumeration.

The integration creates a hybrid computational system where:
- B-Series act as optimization Agents
- P-Systems create computational Arenas (membrane environments)
- J-Surfaces provide differential Relations (dynamic connections)

All components are coordinated through OEIS A000081 rooted tree enumeration.
"""

import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.bpj_integration import create_bpj_system, oeis_a000081

def main():
    print("🚀 BPJ Integration Example: Agent-Arena-Relation Pattern")
    print("=" * 60)
    
    # Display OEIS A000081 sequence
    print("\n📊 OEIS A000081 (Rooted Trees) Enumeration:")
    sequence = oeis_a000081(10)
    print(f"First 10 terms: {sequence}")
    print("This sequence coordinates the BPJ subsystem elements")
    
    # Create synthetic time series data
    print("\n🔬 Generating synthetic time series data...")
    np.random.seed(42)
    
    # Create a simple nonlinear time series
    t = np.linspace(0, 4*np.pi, 200)
    signal = np.sin(t) + 0.5*np.sin(3*t) + 0.2*np.random.randn(200)
    
    # Create input-output pairs for prediction
    window_size = 5
    X = []
    y = []
    for i in range(len(signal) - window_size):
        X.append(signal[i:i+window_size])
        y.append([signal[i+window_size]])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training data: {X_train.shape} -> {y_train.shape}")
    print(f"Test data: {X_test.shape} -> {y_test.shape}")
    
    # Create BPJ integration system
    print("\n🧬 Creating BPJ Integration System...")
    num_elements = 3  # Use 3 BPJ triadic elements
    system = create_bpj_system(
        input_dim=window_size, 
        output_dim=1, 
        num_elements=num_elements
    )
    
    # Display system information
    oeis_info = system.get_oeis_info()
    print(f"Number of BPJ elements: {oeis_info['num_elements']}")
    print(f"OEIS sequence used: {oeis_info['sequence'][:num_elements]}")
    print(f"Each element combines:")
    print("  🤖 B-Series Agent (gradient optimization)")
    print("  🏟️  P-Systems Arena (membrane computing)")
    print("  🔗 J-Surface Relation (differential dynamics)")
    
    # Train the system
    print("\n🎯 Training BPJ System...")
    try:
        system.fit(X_train, y_train)
        print("✅ Training completed successfully!")
        print("   - B-Series Agents optimized gradients")
        print("   - P-Systems Arenas processed membrane dynamics")
        print("   - J-Surface Relations learned differential patterns")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Make predictions
    print("\n🔮 Making Predictions...")
    try:
        predictions = system.predict(X_test)
        print(f"✅ Predictions generated: {predictions.shape}")
        
        # Calculate error metrics
        mse = np.mean((predictions - y_test)**2)
        mae = np.mean(np.abs(predictions - y_test))
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    
    # Demonstrate individual components
    print("\n🔍 Analyzing Individual BPJ Elements...")
    for i, element in enumerate(system.bpj_elements):
        tree_idx = element.tree_index
        print(f"\nElement {i+1} (Tree index {tree_idx}):")
        print(f"  🤖 B-Series: RK order {element.b_config['rk_order']}, ridge {element.b_config['ridge']}")
        print(f"  🏟️  P-Systems: {element.p_config['membranes']} membranes, hierarchical={element.p_config['hierarchical']}")
        print(f"  🔗 J-Surface: {element.j_config['n_reservoir']} neurons, solver='{element.j_config['solver']}'")
    
    print("\n🎉 BPJ Integration Example Completed!")
    print("\nKey achievements:")
    print("✅ Successfully integrated three subsystems with OEIS A000081")
    print("✅ Demonstrated Agent-Arena-Relation pattern")
    print("✅ Achieved functional time series prediction")
    print("✅ Showcased configurable triadic elements")
    
    # Visualize results if matplotlib is available
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot original signal
        plt.subplot(2, 2, 1)
        plt.plot(t, signal)
        plt.title('Original Time Series')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        
        # Plot predictions vs targets
        plt.subplot(2, 2, 2)
        plt.plot(y_test[:50], 'b-', label='True', alpha=0.7)
        plt.plot(predictions[:50], 'r--', label='Predicted', alpha=0.7)
        plt.title('BPJ System Predictions')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        
        # Plot OEIS sequence
        plt.subplot(2, 2, 3)
        oeis_seq = oeis_a000081(15)
        plt.bar(range(1, len(oeis_seq)+1), oeis_seq)
        plt.title('OEIS A000081 (Rooted Trees)')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Count')
        
        # Plot error distribution
        plt.subplot(2, 2, 4)
        errors = predictions.flatten() - y_test.flatten()
        plt.hist(errors, bins=20, alpha=0.7)
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('/tmp/bpj_integration_example.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: /tmp/bpj_integration_example.png")
        
    except ImportError:
        print("\n📊 Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"\n⚠️  Visualization failed: {e}")

if __name__ == "__main__":
    main()