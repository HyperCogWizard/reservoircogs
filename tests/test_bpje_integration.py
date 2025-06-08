"""
Test BPJE Subsystems Integration with OEIS A000081

Tests the enhanced integration functionality of B-Series, P-Systems, J-Surfaces,
and Differential Emotion Theory Framework following the Agent-Arena-Relation-Emotion 
pattern with OEIS A000081 enumeration.
"""

import sys
import os
import numpy as np

def test_bpje_tetradic_element():
    """Test BPJE tetradic element creation and initialization."""
    print("Testing BPJE tetradic element...")
    
    try:
        # Add the repository root to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from reservoirpy.bpj_integration import BPJETetradicElement
        
        # Create a BPJE element
        element = BPJETetradicElement(tree_index=3)
        
        # Check that components are not initialized yet
        if (element.b_agent is not None or element.p_arena is not None or 
            element.j_relation is not None or element.e_emotion is not None):
            print("✗ Components should not be initialized on creation")
            return False
            
        # Initialize with test dimensions
        element.initialize(input_dim=5, output_dim=2)
        
        # Check that all components are now initialized
        if element.b_agent is None:
            print("✗ B-Series agent not initialized")
            return False
        if element.p_arena is None:
            print("✗ P-Systems arena not initialized")
            return False
        if element.j_relation is None:
            print("✗ J-Surface relation not initialized")
            return False
        if element.e_emotion is None:
            print("✗ E-Emotion processor not initialized")
            return False
            
        print("✓ BPJE tetradic element initialization successful")
        return True
        
    except Exception as e:
        print(f"✗ BPJE tetradic element test failed: {e}")
        return False


def test_aare_integration():
    """Test AARE (Agent-Arena-Relation-Emotion) integration system."""
    print("Testing AARE integration system...")
    
    try:
        from reservoirpy.bpj_integration import AARE_Integration
        
        # Create AARE system
        system = AARE_Integration(max_tree_nodes=5)
        
        # Check OEIS sequence is generated
        if len(system.oeis_sequence) == 0:
            print("✗ OEIS sequence not generated")
            return False
            
        # Create ensemble
        system.create_bpje_ensemble(num_elements=3)
        
        if len(system.bpje_elements) != 3:
            print(f"✗ Expected 3 BPJE elements, got {len(system.bpje_elements)}")
            return False
            
        # Initialize ensemble
        system.initialize_ensemble(input_dim=4, output_dim=2)
        
        # Check all elements are initialized
        for i, element in enumerate(system.bpje_elements):
            if not element._initialized:
                print(f"✗ BPJE element {i} not initialized")
                return False
                
        print("✓ AARE integration system working correctly")
        return True
        
    except Exception as e:
        print(f"✗ AARE integration test failed: {e}")
        return False


def test_emotion_processing():
    """Test emotion processing functionality."""
    print("Testing emotion processing...")
    
    try:
        from reservoirpy.bpj_integration import BPJETetradicElement
        
        # Create and initialize element
        element = BPJETetradicElement(tree_index=2)
        element.initialize(input_dim=3, output_dim=1)
        
        # Generate test data for training and testing
        X_train = np.random.randn(5, 3)
        y_train = np.random.randn(5, 1)
        X_test = np.random.randn(3, 3)
        
        # Train first, then test
        _ = element.process_aare(X_train, y_train)
        
        # Process through AARE pattern for prediction
        output = element.process_aare(X_test)
        
        # Check output shape
        if output.shape[0] != X_test.shape[0]:
            print(f"✗ Output batch size mismatch: {output.shape[0]} vs {X_test.shape[0]}")
            return False
            
        # Check emotion state can be retrieved
        emotion_state = element.get_emotion_state()
        if emotion_state is None:
            print("✗ Emotion state not available")
            return False
            
        # Check emotion state contains expected keys
        expected_keys = ['dominant_emotion', 'valence_arousal', 'emotion_vector', 'tree_index']
        for key in expected_keys:
            if key not in emotion_state:
                print(f"✗ Missing emotion state key: {key}")
                return False
                
        # Check dominant emotion is a tuple
        dominant = emotion_state['dominant_emotion']
        if not isinstance(dominant, tuple) or len(dominant) != 2:
            print("✗ Dominant emotion should be (label, intensity) tuple")
            return False
            
        print("✓ Emotion processing working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Emotion processing test failed: {e}")
        return False


def test_ensemble_emotions():
    """Test ensemble emotion state collection."""
    print("Testing ensemble emotion states...")
    
    try:
        from reservoirpy.bpj_integration import create_bpje_system
        
        # Create BPJE system
        system = create_bpje_system(input_dim=4, output_dim=2, num_elements=2)
        
        # Generate training and test data
        X_train = np.random.randn(5, 4)
        y_train = np.random.randn(5, 2)
        X_test = np.random.randn(3, 4)
        
        # Train first
        system.fit(X_train, y_train)
        
        # Then predict to populate emotion states
        _ = system.predict(X_test)
        
        # Get ensemble emotion states
        emotion_states = system.get_ensemble_emotion_states()
        
        if len(emotion_states) != 2:
            print(f"✗ Expected 2 emotion states, got {len(emotion_states)}")
            return False
            
        # Check each emotion state
        for i, state in enumerate(emotion_states):
            if 'tree_index' not in state:
                print(f"✗ Emotion state {i} missing tree_index")
                return False
            if 'dominant_emotion' not in state:
                print(f"✗ Emotion state {i} missing dominant_emotion")
                return False
                
        print("✓ Ensemble emotion states working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Ensemble emotion states test failed: {e}")
        return False


def test_backward_compatibility():
    """Test that old BPJ functionality still works with new BPJE system."""
    print("Testing backward compatibility...")
    
    try:
        # Test old imports still work
        from reservoirpy.bpj_integration import BPJTriadicElement, AAR_Integration, create_bpj_system
        
        # Test old function creates new system
        system = create_bpj_system(input_dim=3, output_dim=1, num_elements=2)
        
        # Check it's actually the new AARE system
        if not hasattr(system, 'bpje_elements'):
            print("✗ Backward compatibility system missing bpje_elements")
            return False
            
        # Check backward compatibility property
        if len(system.bpj_elements) != 2:
            print(f"✗ Backward compatibility property failed: {len(system.bpj_elements)} elements")
            return False
            
        # Test old method names work
        system.create_bpj_ensemble(num_elements=1)
        if len(system.bpje_elements) != 1:
            print("✗ create_bpj_ensemble backward compatibility failed")
            return False
            
        print("✓ Backward compatibility working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False


def test_oeis_enumeration_alignment():
    """Test that emotion dimensions align with OEIS A000081 enumeration."""
    print("Testing OEIS enumeration alignment...")
    
    try:
        from reservoirpy.bpj_integration import BPJETetradicElement, oeis_a000081
        
        oeis_seq = oeis_a000081(8)
        
        # Test different tree indices have different configurations
        elements = []
        for i in range(1, 6):  # Test first 5 elements
            element = BPJETetradicElement(tree_index=i)
            elements.append(element)
            
        # Check that configurations vary based on tree index
        emotion_dims = [elem.e_config['emotion_dimensions'] for elem in elements]
        if len(set(emotion_dims)) == 1:
            print("✗ All emotion dimensions are the same - no OEIS alignment")
            return False
            
        # Check valence_arousal alternates
        valence_settings = [elem.e_config['valence_arousal'] for elem in elements]
        if all(v == valence_settings[0] for v in valence_settings):
            print("✗ All valence_arousal settings are the same - no alternation")
            return False
            
        print("✓ OEIS enumeration alignment working correctly")
        return True
        
    except Exception as e:
        print(f"✗ OEIS enumeration alignment test failed: {e}")
        return False


def test_training_and_prediction():
    """Test BPJE system training and prediction."""
    print("Testing BPJE training and prediction...")
    
    try:
        from reservoirpy.bpj_integration import create_bpje_system
        
        # Create system
        system = create_bpje_system(input_dim=4, output_dim=2, num_elements=2)
        
        # Generate training data
        X_train = np.random.randn(10, 4)
        y_train = np.random.randn(10, 2)
        
        # Train the system
        system.fit(X_train, y_train)
        
        # Generate test data
        X_test = np.random.randn(5, 4)
        
        # Make predictions
        predictions = system.predict(X_test)
        
        # Check prediction shape
        if predictions.shape != (5, 2):
            print(f"✗ Prediction shape mismatch: {predictions.shape} vs (5, 2)")
            return False
            
        # Check predictions are finite
        if not np.all(np.isfinite(predictions)):
            print("✗ Predictions contain non-finite values")
            return False
            
        print("✓ BPJE training and prediction working correctly")
        return True
        
    except Exception as e:
        print(f"✗ BPJE training and prediction test failed: {e}")
        return False


def run_bpje_tests():
    """Run all BPJE integration tests."""
    print("=== BPJE Subsystems Integration Tests ===\n")
    
    tests = [
        test_bpje_tetradic_element,
        test_aare_integration,
        test_emotion_processing,
        test_ensemble_emotions,
        test_backward_compatibility,
        test_oeis_enumeration_alignment,
        test_training_and_prediction,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All BPJE integration tests passed!")
        print("\nB-Series, P-Systems, J-Surfaces, and Differential Emotion Theory")
        print("are successfully integrated with OEIS A000081 enumeration and AARE pattern.")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_bpje_tests()
    sys.exit(0 if success else 1)