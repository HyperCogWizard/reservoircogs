"""
BPJE Integration Example: Demonstrating OEIS A000081 Enumeration 
with P-Systems Membrane Computing, B-Series Rooted Tree Gradient Descent,
J-Surface Julia Differential Equations, and Differential Emotion Theory Framework

This example shows how all four subsystems are linked at the fundamental level
according to rooted tree enumeration and integrated through the 
Agent-Arena-Relation-Emotion (AARE) pattern.
"""

import numpy as np
import sys
import os

# Add the repository root to the path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reservoirpy.bpj_integration import (
    oeis_a000081, 
    create_bpje_system,
    BPJETetradicElement,
    AARE_Integration
)

def demonstrate_oeis_enumeration():
    """Demonstrate OEIS A000081 enumeration underlying the integration."""
    print("=== OEIS A000081 Rooted Tree Enumeration ===")
    
    sequence = oeis_a000081(10)
    print(f"OEIS A000081 sequence (first 10 terms): {sequence}")
    print(f"Description: Number of rooted trees with n nodes")
    print(f"This enumeration provides the fundamental structure for all subsystems\n")
    
    return sequence

def demonstrate_individual_subsystems():
    """Demonstrate how each subsystem is configured according to OEIS enumeration."""
    print("=== Individual Subsystem Configuration (OEIS-aligned) ===")
    
    # Create elements with different tree indices
    for tree_idx in range(1, 6):
        element = BPJETetradicElement(tree_index=tree_idx)
        
        print(f"Tree Index {tree_idx} (OEIS enumeration position):")
        print(f"  B-Series: RK order {element.b_config['rk_order']}, "
              f"ridge {element.b_config['ridge']:.3f}")
        print(f"  P-Systems: {element.p_config['membranes']} membranes, "
              f"hierarchical={element.p_config['hierarchical']}")
        print(f"  J-Surface: spectral_radius {element.j_config['spectral_radius']:.2f}, "
              f"solver {element.j_config['solver']}")
        print(f"  E-Emotion: {element.e_config['emotion_dimensions']} dimensions, "
              f"valence_arousal={element.e_config['valence_arousal']}")
        print()
    
    print("Each subsystem's parameters are uniquely determined by the tree index,")
    print("ensuring structural linkage at the fundamental enumeration level.\n")

def demonstrate_aare_integration():
    """Demonstrate the Agent-Arena-Relation-Emotion integration pattern."""
    print("=== AARE Integration Pattern ===")
    
    # Create a BPJE system
    system = create_bpje_system(input_dim=5, output_dim=3, num_elements=3)
    
    print(f"Created AARE system with {len(system.bpje_elements)} BPJE elements")
    print(f"OEIS sequence: {system.oeis_sequence}")
    
    # Generate sample data
    X_train = np.random.randn(20, 5)
    y_train = np.random.randn(20, 3)
    X_test = np.random.randn(10, 5)
    
    print("\nTraining the AARE ensemble...")
    system.fit(X_train, y_train)
    
    print("Making predictions...")
    predictions = system.predict(X_test)
    print(f"Prediction shape: {predictions.shape}")
    
    # Demonstrate emotion states
    emotion_states = system.get_ensemble_emotion_states()
    print(f"\nEmotion states from {len(emotion_states)} elements:")
    
    for i, state in enumerate(emotion_states):
        dominant_emotion, intensity = state['dominant_emotion']
        valence_arousal = state['valence_arousal']
        print(f"  Element {i+1} (tree {state['tree_index']}): "
              f"Dominant emotion: {dominant_emotion} ({intensity:.3f})")
        if valence_arousal:
            print(f"    Valence: {valence_arousal[0]:.3f}, Arousal: {valence_arousal[1]:.3f}")
    
    return system

def demonstrate_fundamental_linkage():
    """Demonstrate how all subcomponents are linked at the fundamental level."""
    print("\n=== Fundamental Level Linkage ===")
    
    # Create a single element to examine internal linkage
    element = BPJETetradicElement(tree_index=4)
    element.initialize(input_dim=6, output_dim=2)
    
    print(f"Tree Index 4 Element (OEIS position):")
    print(f"  P-Systems membranes: {element.p_config['membranes']}")
    print(f"  Emotion dimensions: {element.e_config['emotion_dimensions']}")
    print(f"  B-Series RK order: {element.b_config['rk_order']}")
    print(f"  J-Surface reservoir size: {element.j_config['n_reservoir']}")
    
    # Show processing flow
    X_sample = np.random.randn(5, 6)
    y_sample = np.random.randn(5, 2)
    
    print(f"\nProcessing flow through AARE pattern:")
    print(f"1. Input shape: {X_sample.shape}")
    
    # Process to see intermediate shapes (for demonstration)
    dummy_arena = element.p_arena(X_sample[:1])  # Single sample for shape
    print(f"2. Arena (P-Systems) output shape: {dummy_arena.shape}")
    
    dummy_emotion = element.e_emotion.forward(dummy_arena)
    print(f"3. Emotion processing output length: {len(dummy_emotion)}")
    
    combined_shape = (dummy_arena.shape[1] + len(dummy_emotion),)
    print(f"4. Combined Arena+Emotion shape: {combined_shape}")
    print(f"5. This feeds into both Relation (J-Surface) and Agent (B-Series)")
    
    # Full processing
    output = element.process_aare(X_sample, y_sample)
    print(f"6. Final integrated output shape: {output.shape}")
    
    print(f"\nThe fundamental linkage ensures:")
    print(f"- Rooted tree branch structure → P-Systems membrane hierarchy")
    print(f"- Elementary differential terms → J-Surface ODE components") 
    print(f"- Emotion dimensions → Fundamental enumeration structure")
    print(f"- B-Series optimization → Coordinated across all components")

def demonstrate_backward_compatibility():
    """Demonstrate backward compatibility with original BPJ system."""
    print("\n=== Backward Compatibility ===")
    
    # Import old names
    from reservoirpy.bpj_integration import BPJTriadicElement, AAR_Integration, create_bpj_system
    
    print("Using original BPJ function names...")
    old_system = create_bpj_system(input_dim=4, output_dim=2, num_elements=2)
    
    print(f"create_bpj_system() created system with {len(old_system.bpj_elements)} elements")
    print(f"System type: {type(old_system).__name__}")
    print(f"Integration type: {old_system.get_oeis_info()['integration_type']}")
    
    # Show it's actually the enhanced system
    X_test = np.random.randn(5, 4)
    y_test = np.random.randn(5, 2)
    old_system.fit(X_test, y_test)
    
    # Get emotion states (new functionality through old interface)
    emotion_states = old_system.get_ensemble_emotion_states()
    print(f"Emotion states available through backward compatible interface: {len(emotion_states)} states")
    
    print("✓ Full backward compatibility maintained while adding emotion framework")

def main():
    """Run the complete BPJE integration demonstration."""
    print("BPJE Subsystems Integration with OEIS A000081 Enumeration")
    print("=" * 60)
    print("Demonstrating integration of:")
    print("- B-Series Rooted Tree Gradient Descent (Agents)")
    print("- P-Systems Membrane Computing (Arenas)")  
    print("- J-Surface Julia Differential Equations (Relations)")
    print("- Differential Emotion Theory Framework (Emotions)")
    print("=" * 60)
    
    # Run demonstrations
    sequence = demonstrate_oeis_enumeration()
    demonstrate_individual_subsystems()
    system = demonstrate_aare_integration()
    demonstrate_fundamental_linkage()
    demonstrate_backward_compatibility()
    
    print("\n" + "=" * 60)
    print("🎉 BPJE Integration Demonstration Complete!")
    print("All subsystems successfully integrated with OEIS A000081 enumeration")
    print("and linked at the most fundamental level through the AARE pattern.")
    print("=" * 60)

if __name__ == "__main__":
    main()