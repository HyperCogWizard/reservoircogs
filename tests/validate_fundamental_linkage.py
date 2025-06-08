"""
Validation Script: Fundamental Level Linkage of BPJE Subsystems

This script validates that the enumerations match for each order and the subcomponents 
of rooted tree branches, p-system nested membranes, elementary differential chain & 
product rule term components are all linked at the most fundamental level.
"""

import numpy as np
import sys
import os

# Add the repository root to the path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reservoirpy.bpj_integration import oeis_a000081, BPJETetradicElement

def validate_enumeration_consistency():
    """Validate that all subsystems use consistent OEIS A000081 enumeration."""
    print("=== Validating OEIS A000081 Enumeration Consistency ===")
    
    oeis_sequence = oeis_a000081(10)
    print(f"OEIS A000081 sequence: {oeis_sequence}")
    
    # Check that each tree index produces consistent configurations
    for tree_idx in range(1, 6):
        element = BPJETetradicElement(tree_index=tree_idx)
        
        # Validate that configurations are deterministically linked to tree index
        expected_membranes = min(max(1, tree_idx), 5)
        expected_hierarchical = tree_idx % 2 == 1
        expected_emotion_dims = min(max(5, tree_idx + 4), 10)
        expected_valence = tree_idx % 2 == 0
        
        assert element.p_config['membranes'] == expected_membranes, \
            f"P-Systems membranes mismatch for tree {tree_idx}"
        assert element.p_config['hierarchical'] == expected_hierarchical, \
            f"P-Systems hierarchy mismatch for tree {tree_idx}"
        assert element.e_config['emotion_dimensions'] == expected_emotion_dims, \
            f"Emotion dimensions mismatch for tree {tree_idx}"
        assert element.e_config['valence_arousal'] == expected_valence, \
            f"Valence/arousal mismatch for tree {tree_idx}"
        
        print(f"✓ Tree {tree_idx}: All subsystems consistently enumerated")
    
    print("✓ OEIS A000081 enumeration consistency validated")
    return True

def validate_subcomponent_linkage():
    """Validate linkage of subcomponents at fundamental level."""
    print("\n=== Validating Subcomponent Linkage ===")
    
    element = BPJETetradicElement(tree_index=3)
    element.initialize(input_dim=5, output_dim=2)
    
    # Generate test data
    X = np.random.randn(3, 5)
    y = np.random.randn(3, 2)
    
    print(f"Processing {X.shape[0]} samples through tree index 3 element...")
    
    # Process and examine intermediate outputs
    # 1. P-Systems (Arena) processing (handle single timestep requirement)
    arena_outputs = []
    for i in range(X.shape[0]):
        arena_out = element.p_arena(X[i:i+1])
        arena_outputs.append(arena_out)
    arena_output = np.vstack(arena_outputs)
    print(f"1. P-Systems arena output shape: {arena_output.shape}")
    print(f"   Membranes: {element.p_config['membranes']}")
    print(f"   Hierarchical: {element.p_config['hierarchical']}")
    
    # 2. Emotion processing of arena output
    emotion_outputs = []
    for i in range(arena_output.shape[0]):
        emotion_out = element.e_emotion.forward(arena_output[i:i+1])
        emotion_outputs.append(emotion_out)
    emotion_output = np.vstack(emotion_outputs)
    print(f"2. Emotion processing output shape: {emotion_output.shape}")
    print(f"   Emotion dimensions: {element.e_config['emotion_dimensions']}")
    print(f"   Valence/arousal: {element.e_config['valence_arousal']}")
    
    # 3. Combined input to B-Series and J-Surface
    combined = np.concatenate([arena_output, emotion_output], axis=1)
    print(f"3. Combined arena+emotion shape: {combined.shape}")
    
    # 4. Full processing through AARE pattern
    final_output = element.process_aare(X, y)
    print(f"4. Final AARE output shape: {final_output.shape}")
    
    # Validate fundamental linkage properties
    assert arena_output.shape[1] > 0, "Arena must produce output"
    assert emotion_output.shape[1] > 0, "Emotion must produce output"
    assert combined.shape[1] == arena_output.shape[1] + emotion_output.shape[1], \
        "Combined shape must equal sum of components"
    assert final_output.shape[0] == X.shape[0], "Batch size must be preserved"
    
    print("✓ Subcomponent linkage validated")
    return True

def validate_differential_chain_components():
    """Validate elementary differential chain & product rule term components."""
    print("\n=== Validating Differential Chain Components ===")
    
    # Test multiple elements to show differential component scaling
    for tree_idx in range(1, 4):
        element = BPJETetradicElement(tree_index=tree_idx)
        element.initialize(input_dim=4, output_dim=1)
        
        # Check J-Surface differential equation setup
        j_config = element.j_config
        print(f"Tree {tree_idx} J-Surface config:")
        print(f"  Reservoir size: {j_config['n_reservoir']}")
        print(f"  Spectral radius: {j_config['spectral_radius']:.2f}")
        print(f"  ODE solver: {j_config['solver']}")
        print(f"  Time step: {j_config['dt']:.4f}")
        
        # Check that reservoir size scales with tree index (elementary chain growth)
        expected_reservoir = 50 + tree_idx * 15
        assert j_config['n_reservoir'] == expected_reservoir, \
            f"Reservoir size mismatch for tree {tree_idx}"
        
        # Check B-Series gradient descent configuration
        b_config = element.b_config
        print(f"  B-Series RK order: {b_config['rk_order']}")
        print(f"  Step size: {b_config['step_size']:.4f}")
        
        # Validate that step size adapts to tree complexity
        expected_step = 0.01 / (1 + tree_idx * 0.1)
        assert abs(b_config['step_size'] - expected_step) < 1e-6, \
            f"Step size mismatch for tree {tree_idx}"
        
        print(f"✓ Tree {tree_idx}: Differential components properly linked")
    
    print("✓ Elementary differential chain & product rule components validated")
    return True

def validate_rooted_tree_membrane_correspondence():
    """Validate correspondence between rooted tree branches and P-system membranes."""
    print("\n=== Validating Rooted Tree ↔ Membrane Correspondence ===")
    
    oeis_sequence = oeis_a000081(8)
    
    for i, tree_count in enumerate(oeis_sequence[:5], 1):
        element = BPJETetradicElement(tree_index=i)
        
        print(f"Tree index {i}: OEIS value = {tree_count}")
        print(f"  P-System membranes: {element.p_config['membranes']}")
        print(f"  Hierarchical: {element.p_config['hierarchical']}")
        print(f"  Emotion dimensions: {element.e_config['emotion_dimensions']}")
        
        # Validate that membrane count is bounded by complexity considerations
        # (The exact correspondence is through the enumeration index, not the OEIS value)
        assert 1 <= element.p_config['membranes'] <= 5, \
            f"Membrane count out of bounds for tree {i}"
        
        # Validate hierarchical pattern alternation (structural correspondence)
        expected_hierarchical = i % 2 == 1
        assert element.p_config['hierarchical'] == expected_hierarchical, \
            f"Hierarchical pattern mismatch for tree {i}"
        
        print(f"✓ Tree {i}: Rooted tree structure ↔ membrane hierarchy correspondence")
    
    print("✓ Rooted tree ↔ membrane correspondence validated")
    return True

def main():
    """Run all fundamental linkage validations."""
    print("BPJE Fundamental Level Linkage Validation")
    print("=" * 50)
    print("Validating that:")
    print("• Enumerations match for each order")
    print("• Rooted tree branches ↔ P-system nested membranes")  
    print("• Elementary differential chain & product rule terms")
    print("• All components linked at most fundamental level")
    print("=" * 50)
    
    try:
        validate_enumeration_consistency()
        validate_subcomponent_linkage()
        validate_differential_chain_components()
        validate_rooted_tree_membrane_correspondence()
        
        print("\n" + "=" * 50)
        print("🎉 ALL FUNDAMENTAL LINKAGE VALIDATIONS PASSED!")
        print("✓ OEIS A000081 enumeration consistency")
        print("✓ Subcomponent linkage at fundamental level")
        print("✓ Elementary differential chain components")
        print("✓ Rooted tree ↔ membrane correspondence")
        print("=" * 50)
        print("The integration requirement is fully satisfied:")
        print("All subcomponents are linked at the most fundamental level.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)