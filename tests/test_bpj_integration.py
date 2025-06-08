"""
Test BPJ Subsystems Integration with OEIS A000081

Tests the core integration functionality of B-Series, P-Systems, and J-Surfaces
following the Agent-Arena-Relation pattern with OEIS A000081 enumeration.
"""

import sys
import os
import numpy as np

def test_oeis_a000081():
    """Test OEIS A000081 sequence generation."""
    print("Testing OEIS A000081 sequence generation...")
    
    try:
        # Add the repository root to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from reservoirpy.bpj_integration import oeis_a000081
        
        # Test known values of OEIS A000081
        sequence = oeis_a000081(10)
        expected = [1, 1, 2, 4, 9, 20, 48, 115, 274, 668]
        
        if len(sequence) != len(expected):
            print(f"✗ Sequence length mismatch: got {len(sequence)}, expected {len(expected)}")
            return False
            
        for i, (got, exp) in enumerate(zip(sequence, expected)):
            if got != exp:
                print(f"✗ Sequence mismatch at index {i}: got {got}, expected {exp}")
                return False
                
        print("✓ OEIS A000081 sequence generation is correct")
        return True
        
    except Exception as e:
        print(f"✗ OEIS A000081 test failed: {e}")
        return False


def test_bpj_triadic_element():
    """Test BPJ triadic element creation and initialization."""
    print("Testing BPJ triadic element...")
    
    try:
        from reservoirpy.bpj_integration import BPJTriadicElement
        
        # Create a BPJ element
        element = BPJTriadicElement(tree_index=3)
        
        # Check that components are not initialized yet
        if element.b_agent is not None or element.p_arena is not None or element.j_relation is not None:
            print("✗ Components should not be initialized on creation")
            return False
            
        # Initialize with test dimensions
        element.initialize(input_dim=5, output_dim=2)
        
        # Check that components are now initialized
        if element.b_agent is None:
            print("✗ B-Series agent not initialized")
            return False
        if element.p_arena is None:
            print("✗ P-Systems arena not initialized")
            return False
        if element.j_relation is None:
            print("✗ J-Surface relation not initialized")
            return False
            
        print("✓ BPJ triadic element initialization successful")
        return True
        
    except Exception as e:
        print(f"✗ BPJ triadic element test failed: {e}")
        return False


def test_aar_integration():
    """Test AAR (Agent-Arena-Relation) integration system."""
    print("Testing AAR integration system...")
    
    try:
        from reservoirpy.bpj_integration import AAR_Integration
        
        # Create AAR system
        system = AAR_Integration(max_tree_nodes=5)
        
        # Check OEIS sequence is generated
        if len(system.oeis_sequence) == 0:
            print("✗ OEIS sequence not generated")
            return False
            
        # Create ensemble
        system.create_bpj_ensemble(num_elements=3)
        
        if len(system.bpj_elements) != 3:
            print(f"✗ Expected 3 BPJ elements, got {len(system.bpj_elements)}")
            return False
            
        # Initialize ensemble
        system.initialize_ensemble(input_dim=4, output_dim=2)
        
        # Check all elements are initialized
        for i, element in enumerate(system.bpj_elements):
            if not element._initialized:
                print(f"✗ BPJ element {i} not initialized")
                return False
                
        print("✓ AAR integration system working correctly")
        return True
        
    except Exception as e:
        print(f"✗ AAR integration test failed: {e}")
        return False


def test_bpj_system_creation():
    """Test the convenience function for creating BPJ systems."""
    print("Testing BPJ system creation...")
    
    try:
        from reservoirpy.bpj_integration import create_bpj_system
        
        # Create a complete BPJ system
        system = create_bpj_system(input_dim=3, output_dim=1, num_elements=2)
        
        if not system._initialized:
            print("✗ BPJ system not initialized")
            return False
            
        if len(system.bpj_elements) != 2:
            print(f"✗ Expected 2 BPJ elements, got {len(system.bpj_elements)}")
            return False
            
        # Test OEIS info
        oeis_info = system.get_oeis_info()
        if 'sequence' not in oeis_info:
            print("✗ OEIS info missing sequence")
            return False
            
        print("✓ BPJ system creation successful")
        return True
        
    except Exception as e:
        print(f"✗ BPJ system creation test failed: {e}")
        return False


def test_integration_imports():
    """Test that the integration can be imported from main reservoirpy."""
    print("Testing integration imports...")
    
    try:
        import reservoirpy
        
        # Check that bpj_integration is available
        if not hasattr(reservoirpy, 'bpj_integration'):
            print("✗ bpj_integration not available in reservoirpy")
            return False
            
        # Test importing specific functions
        from reservoirpy.bpj_integration import create_bpj_system, oeis_a000081
        
        # Quick functional test
        sequence = oeis_a000081(3)
        if sequence != [1, 1, 2]:
            print("✗ OEIS function not working through import")
            return False
            
        print("✓ Integration imports working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Integration imports test failed: {e}")
        return False


def test_config_customization():
    """Test that custom configurations are properly applied."""
    print("Testing configuration customization...")
    
    try:
        from reservoirpy.bpj_integration import BPJTriadicElement
        
        # Create element with custom configs
        custom_b_config = {'ridge': 0.5, 'rk_order': 6}
        custom_p_config = {'membranes': 4, 'hierarchical': False}
        custom_j_config = {'spectral_radius': 0.8, 'solver': 'Vern7'}
        
        element = BPJTriadicElement(
            tree_index=1,
            b_series_config=custom_b_config,
            p_systems_config=custom_p_config,
            j_surface_config=custom_j_config
        )
        
        # Check configs are applied
        if element.b_config['ridge'] != 0.5:
            print("✗ Custom B-Series config not applied")
            return False
        if element.p_config['membranes'] != 4:
            print("✗ Custom P-Systems config not applied")
            return False
        if element.j_config['spectral_radius'] != 0.8:
            print("✗ Custom J-Surface config not applied")
            return False
            
        print("✓ Configuration customization working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Configuration customization test failed: {e}")
        return False


def run_integration_tests():
    """Run all BPJ integration tests."""
    print("=== BPJ Subsystems Integration Tests ===\n")
    
    tests = [
        test_oeis_a000081,
        test_bpj_triadic_element,
        test_aar_integration,
        test_bpj_system_creation,
        test_integration_imports,
        test_config_customization,
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
        print("🎉 All BPJ integration tests passed!")
        print("\nB-Series, P-Systems, and J-Surfaces are successfully integrated")
        print("with OEIS A000081 enumeration and AAR pattern.")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)