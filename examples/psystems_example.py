#!/usr/bin/env python3
"""
P-Systems Membrane Computing Example

This example demonstrates the usage of P-Systems Membrane Computing 
for bio-inspired hierarchical processing using membrane-based reservoir partitions.

P-Systems Membrane Computing enables:
- Bio-inspired computational models
- Hierarchical processing through membrane layers
- Parallel processing across multiple membranes
- Future integration with P-lingua rules
"""

import numpy as np
from reservoirpy.experimental import MembraneComputing
from reservoirpy.datasets import mackey_glass


def demo_basic_usage():
    """Demonstrate basic P-Systems Membrane Computing usage."""
    print("=== Basic P-Systems Membrane Computing Demo ===\n")
    
    # Create a membrane computing system with 3 membranes
    membrane_reservoir = MembraneComputing(
        membranes=3,
        hierarchical=True,
        membrane_size=100
    )
    
    # Generate sample data
    X = np.random.randn(50, 5)
    print(f"Input data shape: {X.shape}")
    
    # Process data through membrane system
    output = membrane_reservoir.run(X)
    print(f"Output data shape: {output.shape}")
    
    # Get membrane information
    info = membrane_reservoir.get_membrane_info()
    print(f"Membrane configuration: {info}")
    
    # Get current membrane states
    states = membrane_reservoir.get_membrane_states()
    print(f"Number of membrane states: {len(states)}")
    print()


def demo_hierarchical_vs_parallel():
    """Compare hierarchical vs parallel membrane processing."""
    print("=== Hierarchical vs Parallel Processing Comparison ===\n")
    
    # Generate test data
    X = np.random.randn(100, 10)
    
    # Hierarchical processing
    hierarchical_system = MembraneComputing(
        membranes=4,
        hierarchical=True,
        membrane_size=50
    )
    hierarchical_output = hierarchical_system.run(X)
    
    # Parallel processing  
    parallel_system = MembraneComputing(
        membranes=4,
        hierarchical=False,
        membrane_size=50
    )
    parallel_output = parallel_system.run(X)
    
    print(f"Input shape: {X.shape}")
    print(f"Hierarchical output shape: {hierarchical_output.shape}")
    print(f"Parallel output shape: {parallel_output.shape}")
    print()
    
    return hierarchical_output, parallel_output


def demo_mackey_glass_processing():
    """Demonstrate P-Systems processing on Mackey-Glass time series."""
    print("=== Mackey-Glass Time Series Processing ===\n")
    
    # Generate Mackey-Glass time series
    mg_data = mackey_glass(2000)
    
    # Create membrane computing system
    membrane_system = MembraneComputing(
        membranes=5,
        hierarchical=True,
        membrane_size=80
    )
    
    # Process the time series
    processed_output = membrane_system.run(mg_data)
    
    print(f"Mackey-Glass data shape: {mg_data.shape}")
    print(f"Processed output shape: {processed_output.shape}")
    
    # Calculate some basic statistics
    print(f"Input mean: {np.mean(mg_data):.4f}, std: {np.std(mg_data):.4f}")
    print(f"Output mean: {np.mean(processed_output):.4f}, std: {np.std(processed_output):.4f}")
    print()
    
    return mg_data, processed_output


def demo_membrane_configuration():
    """Demonstrate different membrane configurations."""
    print("=== Different Membrane Configurations ===\n")
    
    configurations = [
        (1, True, 50),   # Single membrane
        (2, True, 30),   # Small hierarchical
        (3, False, 40),  # Medium parallel
        (6, True, 20),   # Large hierarchical
    ]
    
    X = np.random.randn(50, 8)
    
    for membranes, hierarchical, size in configurations:
        system = MembraneComputing(
            membranes=membranes,
            hierarchical=hierarchical,
            membrane_size=size
        )
        
        output = system.run(X)
        mode = "Hierarchical" if hierarchical else "Parallel"
        
        print(f"{membranes} membranes, {mode}, size {size}: "
              f"Input {X.shape} -> Output {output.shape}")
    
    print()


def demo_p_lingua_integration():
    """Demonstrate P-lingua rules integration framework."""
    print("=== P-lingua Rules Integration Framework ===\n")
    
    # Create system with P-lingua rules placeholder
    membrane_system = MembraneComputing(
        membranes=3,
        hierarchical=True,
        membrane_size=60,
        p_lingua_rules="example_rules.pl"  # Future: actual P-lingua file
    )
    
    print(f"Initial P-lingua rules: {membrane_system.p_lingua_rules}")
    
    # Update P-lingua rules
    membrane_system.set_p_lingua_rules("advanced_rules.pl")
    print(f"Updated P-lingua rules: {membrane_system.hypers['p_lingua_rules']}")
    
    # Process some data
    X = np.random.randn(30, 6)
    output = membrane_system.run(X)
    
    print(f"Processing with P-lingua framework: {X.shape} -> {output.shape}")
    print("Note: P-lingua rule parsing and application is planned for future versions")
    print()


if __name__ == "__main__":
    print("P-Systems Membrane Computing Examples")
    print("=====================================\n")
    
    # Run all demonstrations
    demo_basic_usage()
    demo_hierarchical_vs_parallel()
    demo_mackey_glass_processing()
    demo_membrane_configuration()
    demo_p_lingua_integration()
    
    print("=== Summary ===")
    print("P-Systems Membrane Computing provides:")
    print("• Bio-inspired computational models")
    print("• Hierarchical and parallel processing modes")
    print("• Flexible membrane configurations")
    print("• Framework for future P-lingua integration")
    print("• Seamless integration with ReservoirPy ecosystem")
    print("\nThis implementation serves as the foundation for the roadmap feature")
    print("described in issue #11 for Q1 2025 development.")