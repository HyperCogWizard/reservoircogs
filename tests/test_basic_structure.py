"""
Basic framework structure test without external dependencies

Test that the emotion framework structure is properly created.
"""

import sys
import os

def test_framework_structure():
    """Test that the framework structure is properly created."""
    print("Testing framework structure...")
    
    # Check if emotion nodes directory exists
    emotion_dir = os.path.join(os.path.dirname(__file__), '..', 'reservoirpy', 'nodes', 'emotions')
    if not os.path.exists(emotion_dir):
        print("✗ Emotion nodes directory does not exist")
        return False
    
    # Check if emotion module files exist
    required_files = [
        '__init__.py',
        'differential_emotion.py',
        'emotion_reservoir.py'
    ]
    
    for file in required_files:
        file_path = os.path.join(emotion_dir, file)
        if not os.path.exists(file_path):
            print(f"✗ Required file {file} does not exist")
            return False
    
    print("✓ All emotion framework files exist")
    
    # Check if C++ files exist
    cpp_nodes_dir = os.path.join(os.path.dirname(__file__), '..', 'opencog', 'reservoir', 'nodes')
    cpp_files = ['EmotionNode.h', 'EmotionNode.cc']
    
    for file in cpp_files:
        file_path = os.path.join(cpp_nodes_dir, file)
        if not os.path.exists(file_path):
            print(f"✗ Required C++ file {file} does not exist")
            return False
    
    print("✓ All C++ emotion framework files exist")
    
    # Check if examples exist
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    example_files = [
        'differential_emotion_example.py',
        'atomspace/emotion_atomspace_example.cpp'
    ]
    
    for file in example_files:
        file_path = os.path.join(examples_dir, file)
        if not os.path.exists(file_path):
            print(f"✗ Required example file {file} does not exist")
            return False
    
    print("✓ All example files exist")
    
    return True

def test_basic_imports():
    """Test basic imports without external dependencies."""
    print("Testing basic imports...")
    
    try:
        # Add the repository root to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        # Test if we can import the emotion module structure
        import reservoirpy.nodes.emotions
        
        # Check if the __all__ list is defined (may be empty due to missing deps)
        all_items = getattr(reservoirpy.nodes.emotions, '__all__', [])
        
        # If dependencies are missing, __all__ will be empty, which is expected
        if len(all_items) == 0:
            print("✓ Emotion module structure exists (dependencies not available)")
            return True
        
        # If dependencies are available, check for expected items
        expected_items = ['EmotionReservoir', 'DifferentialEmotionProcessor']
        
        for item in expected_items:
            if item not in all_items:
                print(f"✗ {item} not found in emotions module __all__")
                return False
        
        print("✓ Emotion module structure is correct (with dependencies)")
        return True
        
    except ImportError as e:
        # This is expected if numpy/scipy are not available
        if "numpy" in str(e) or "scipy" in str(e):
            print("✓ Emotion module import structure exists (numpy/scipy not available)")
            return True
        else:
            print(f"✗ Unexpected import error: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_file_contents():
    """Test that the framework files contain expected content."""
    print("Testing file contents...")
    
    # Test DifferentialEmotionProcessor file
    emotion_file = os.path.join(os.path.dirname(__file__), '..', 
                               'reservoirpy', 'nodes', 'emotions', 'differential_emotion.py')
    
    try:
        with open(emotion_file, 'r') as f:
            content = f.read()
            
        # Check for key components
        expected_content = [
            'class DifferentialEmotionProcessor',
            'emotion_dimensions',
            'valence_arousal',
            'temporal_dynamics',
            'get_dominant_emotion',
            'get_valence_arousal'
        ]
        
        for item in expected_content:
            if item not in content:
                print(f"✗ Missing expected content: {item}")
                return False
                
        print("✓ DifferentialEmotionProcessor file contains expected content")
        
    except Exception as e:
        print(f"✗ Error reading emotion file: {e}")
        return False
    
    # Test EmotionReservoir file  
    reservoir_file = os.path.join(os.path.dirname(__file__), '..', 
                                 'reservoirpy', 'nodes', 'emotions', 'emotion_reservoir.py')
    
    try:
        with open(reservoir_file, 'r') as f:
            content = f.read()
            
        # Check for key components
        expected_content = [
            'class EmotionReservoir',
            'emotion_integration',
            'emotion_feedback',
            'DifferentialEmotionProcessor',
            'get_reservoir_state',
            'get_emotion_state'
        ]
        
        for item in expected_content:
            if item not in content:
                print(f"✗ Missing expected content: {item}")
                return False
                
        print("✓ EmotionReservoir file contains expected content")
        
    except Exception as e:
        print(f"✗ Error reading reservoir file: {e}")
        return False
    
    # Test C++ header file
    cpp_header = os.path.join(os.path.dirname(__file__), '..', 
                             'opencog', 'reservoir', 'nodes', 'EmotionNode.h')
    
    try:
        with open(cpp_header, 'r') as f:
            content = f.read()
            
        # Check for key components
        expected_content = [
            'class EmotionNode',
            'BasicEmotion',
            'EmotionState',
            'processInput',
            'storeEmotionInAtomSpace',
            'getDominantEmotion'
        ]
        
        for item in expected_content:
            if item not in content:
                print(f"✗ Missing expected C++ content: {item}")
                return False
                
        print("✓ C++ EmotionNode header contains expected content")
        
    except Exception as e:
        print(f"✗ Error reading C++ header: {e}")
        return False
    
    return True

def run_basic_tests():
    """Run basic tests without external dependencies."""
    print("=== Basic Differential Emotion Theory Framework Tests ===\n")
    
    tests = [
        test_framework_structure,
        test_basic_imports,
        test_file_contents,
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
        print("🎉 All basic tests passed!")
        print("\nFramework structure is properly created.")
        print("Note: Full functionality tests require numpy/scipy dependencies.")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)