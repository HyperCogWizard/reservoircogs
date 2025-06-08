"""
Tests for Differential Emotion Theory Framework

Basic tests to ensure the emotion framework components work correctly.
"""

import numpy as np
import sys
import os

# Add the repository root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_differential_emotion_processor():
    """Test the basic functionality of DifferentialEmotionProcessor."""
    try:
        from reservoirpy.nodes.emotions import DifferentialEmotionProcessor
        
        print("Testing DifferentialEmotionProcessor...")
        
        # Create processor
        processor = DifferentialEmotionProcessor(
            emotion_dimensions=10,
            valence_arousal=True,
            temporal_dynamics=True
        )
        
        # Test initialization
        test_input = np.array([0.5, 0.3, 0.2])
        processor.initialize(test_input)
        
        # Test forward pass
        output = processor.forward(test_input)
        
        # Verify output shape
        expected_shape = 10 + 2  # 10 emotions + valence + arousal
        assert len(output) == expected_shape, f"Expected output shape {expected_shape}, got {len(output)}"
        
        # Test emotion bounds (should be non-negative and sum to 1 for basic emotions)
        basic_emotions = output[:10]
        assert all(e >= 0 for e in basic_emotions), "Basic emotions should be non-negative"
        assert abs(np.sum(basic_emotions) - 1.0) < 1e-6, "Basic emotions should sum to 1"
        
        # Test valence and arousal (should be in reasonable range)
        valence, arousal = output[-2], output[-1]
        assert -2 <= valence <= 2, f"Valence {valence} out of expected range [-2, 2]"
        assert -2 <= arousal <= 2, f"Arousal {arousal} out of expected range [-2, 2]"
        
        # Test dominant emotion
        dominant_emotion, intensity = processor.get_dominant_emotion()
        assert isinstance(dominant_emotion, str), "Dominant emotion should be a string"
        assert 0 <= intensity <= 1, f"Emotion intensity {intensity} should be in [0, 1]"
        
        # Test valence/arousal getter
        va = processor.get_valence_arousal()
        assert va is not None, "Valence/arousal should not be None"
        assert len(va) == 2, "Valence/arousal should return tuple of length 2"
        
        # Test emotion vector
        emotion_vector = processor.get_emotion_vector()
        assert isinstance(emotion_vector, dict), "Emotion vector should be a dictionary"
        assert len(emotion_vector) == 12, "Should have 10 emotions + valence + arousal"
        
        print("✓ DifferentialEmotionProcessor tests passed")
        return True
        
    except Exception as e:
        print(f"✗ DifferentialEmotionProcessor test failed: {e}")
        return False

def test_emotion_reservoir():
    """Test the basic functionality of EmotionReservoir."""
    try:
        from reservoirpy.nodes.emotions import EmotionReservoir
        
        print("Testing EmotionReservoir...")
        
        # Create emotion reservoir
        reservoir = EmotionReservoir(
            units=50,
            emotion_integration=0.1,
            emotion_dimensions=10,
            emotion_feedback=True,
            lr=0.3,
            sr=0.9
        )
        
        # Test initialization
        test_input = np.array([0.5, 0.3, 0.2])
        reservoir.initialize(test_input)
        
        # Test forward pass
        output = reservoir.forward(test_input)
        
        # Verify output shape (reservoir units + emotions + valence/arousal)
        expected_shape = 50 + 10 + 2  # 50 units + 10 emotions + valence + arousal
        assert len(output) == expected_shape, f"Expected output shape {expected_shape}, got {len(output)}"
        
        # Test reservoir state getter
        reservoir_state = reservoir.get_reservoir_state()
        assert len(reservoir_state) == 50, "Reservoir state should have 50 units"
        
        # Test emotion state getter
        emotion_state = reservoir.get_emotion_state()
        assert len(emotion_state) == 12, "Emotion state should have 12 dimensions"
        
        # Test dominant emotion
        dominant_emotion = reservoir.get_dominant_emotion()
        assert isinstance(dominant_emotion, tuple), "Dominant emotion should be a tuple"
        assert len(dominant_emotion) == 2, "Dominant emotion should have name and intensity"
        
        # Test valence/arousal
        valence_arousal = reservoir.get_valence_arousal()
        assert valence_arousal is not None, "Valence/arousal should not be None"
        assert len(valence_arousal) == 2, "Valence/arousal should be tuple of length 2"
        
        # Test emotion labels
        labels = reservoir.get_emotion_labels()
        assert len(labels) == 12, "Should have 12 emotion labels"
        assert "joy" in labels, "Should include 'joy' emotion"
        assert "valence" in labels, "Should include 'valence'"
        assert "arousal" in labels, "Should include 'arousal'"
        
        # Test emotion integration setting
        reservoir.set_emotion_integration(0.2)
        assert reservoir.emotion_integration == 0.2, "Emotion integration should be updated"
        
        # Test emotion state reset
        reservoir.reset_emotion_state()
        emotion_state_after_reset = reservoir.get_emotion_state()
        assert np.allclose(emotion_state_after_reset, 0), "Emotion state should be reset to zero"
        
        print("✓ EmotionReservoir tests passed")
        return True
        
    except Exception as e:
        print(f"✗ EmotionReservoir test failed: {e}")
        return False

def test_temporal_dynamics():
    """Test temporal emotion dynamics."""
    try:
        from reservoirpy.nodes.emotions import DifferentialEmotionProcessor
        
        print("Testing temporal dynamics...")
        
        processor = DifferentialEmotionProcessor(
            emotion_dimensions=5,  # Smaller for easier testing
            valence_arousal=False,  # Simpler for testing
            temporal_dynamics=True
        )
        
        # Initialize with first input
        input1 = np.array([1.0, 0.0, 0.0])
        processor.initialize(input1)
        state1 = processor.forward(input1)
        
        # Process second input (should show temporal influence)
        input2 = np.array([0.0, 1.0, 0.0])
        state2 = processor.forward(input2)
        
        # The second state should be influenced by the first due to temporal dynamics
        # It shouldn't be exactly what we'd get from input2 alone
        processor_no_temporal = DifferentialEmotionProcessor(
            emotion_dimensions=5,
            valence_arousal=False,
            temporal_dynamics=False
        )
        processor_no_temporal.initialize(input2)
        state2_no_temporal = processor_no_temporal.forward(input2)
        
        # States should be different due to temporal dynamics
        assert not np.allclose(state2, state2_no_temporal, atol=1e-6), \
            "Temporal dynamics should create different states"
        
        print("✓ Temporal dynamics tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Temporal dynamics test failed: {e}")
        return False

def test_imports():
    """Test that all emotion modules can be imported correctly."""
    try:
        print("Testing imports...")
        
        # Test individual imports
        from reservoirpy.nodes.emotions import DifferentialEmotionProcessor
        from reservoirpy.nodes.emotions import EmotionReservoir
        
        # Test import from main nodes module
        from reservoirpy.nodes import DifferentialEmotionProcessor as DEP
        from reservoirpy.nodes import EmotionReservoir as ER
        
        # Verify they're the same classes
        assert DifferentialEmotionProcessor is DEP, "Import from emotions vs nodes should be same class"
        assert EmotionReservoir is ER, "Import from emotions vs nodes should be same class"
        
        print("✓ Import tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("=== Differential Emotion Theory Framework Tests ===\n")
    
    tests = [
        test_imports,
        test_differential_emotion_processor,
        test_emotion_reservoir,
        test_temporal_dynamics,
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
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)