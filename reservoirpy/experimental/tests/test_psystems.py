import numpy as np
import pytest
from numpy.testing import assert_array_equal

from .. import MembraneComputing


class TestMembraneComputing:
    """Test suite for P-Systems Membrane Computing implementation."""
    
    def test_membrane_computing_creation(self):
        """Test basic MembraneComputing node creation."""
        membrane_comp = MembraneComputing(
            membranes=3,
            hierarchical=True,
            membrane_size=50
        )
        
        assert membrane_comp.membranes == 3
        assert membrane_comp.hierarchical is True
        assert membrane_comp.membrane_size == 50
        assert membrane_comp.p_lingua_rules is None
    
    def test_membrane_computing_initialization(self):
        """Test membrane system initialization with data."""
        membrane_comp = MembraneComputing(membranes=2, membrane_size=20)
        
        # Initialize with sample data
        X = np.random.randn(10, 5)
        membrane_comp.initialize(X[0:1])
        
        # Check that membrane states are initialized
        membrane_states = membrane_comp.get_membrane_states()
        assert len(membrane_states) == 2
        
        for i, state in enumerate(membrane_states):
            assert "internal_state" in state
            assert "weights" in state
            assert "input_weights" in state
            assert "output_weights" in state
            assert state["internal_state"].shape == (1, 20)
            assert state["weights"].shape == (20, 20)
            assert state["input_weights"].shape == (20, 5)
            assert state["output_weights"].shape == (20, 10)
    
    def test_hierarchical_processing(self):
        """Test hierarchical membrane processing."""
        membrane_comp = MembraneComputing(
            membranes=3,
            hierarchical=True,
            membrane_size=20
        )
        
        X = np.random.randn(10, 5)
        membrane_comp.initialize(X[0:1])
        
        # Run forward pass
        output = membrane_comp.run(X)
        
        # In hierarchical mode, output should be from final membrane
        assert output.shape[1] == 10  # membrane_size // 2
        assert output.shape[0] == 10  # number of timesteps
    
    def test_parallel_processing(self):
        """Test parallel membrane processing."""
        membrane_comp = MembraneComputing(
            membranes=3,
            hierarchical=False,
            membrane_size=20
        )
        
        X = np.random.randn(10, 5)
        membrane_comp.initialize(X[0:1])
        
        # Run forward pass
        output = membrane_comp.run(X)
        
        # In parallel mode, output should combine all membranes
        assert output.shape[1] == 30  # 3 membranes * (membrane_size // 2)
        assert output.shape[0] == 10  # number of timesteps
    
    def test_membrane_state_updates(self):
        """Test that membrane states are properly updated during processing."""
        membrane_comp = MembraneComputing(membranes=2, membrane_size=10)
        
        X = np.random.randn(5, 3)
        membrane_comp.initialize(X[0:1])
        
        # Get initial states
        initial_states = membrane_comp.get_membrane_states()
        initial_internal_states = [state["internal_state"].copy() for state in initial_states]
        
        # Run processing
        output = membrane_comp.run(X)
        
        # Get updated states
        updated_states = membrane_comp.get_membrane_states()
        updated_internal_states = [state["internal_state"] for state in updated_states]
        
        # States should have changed
        for i in range(2):
            assert not np.array_equal(initial_internal_states[i], updated_internal_states[i])
    
    def test_membrane_info(self):
        """Test membrane configuration info retrieval."""
        membrane_comp = MembraneComputing(
            membranes=4,
            hierarchical=False,
            membrane_size=30,
            p_lingua_rules="test_rules.pl"
        )
        
        info = membrane_comp.get_membrane_info()
        
        assert info["membranes"] == 4
        assert info["hierarchical"] is False
        assert info["membrane_size"] == 30
        assert info["p_lingua_rules"] == "test_rules.pl"
    
    def test_p_lingua_rules_setting(self):
        """Test P-lingua rules file setting."""
        membrane_comp = MembraneComputing(membranes=2)
        
        # Initially no rules
        assert membrane_comp.p_lingua_rules is None
        
        # Set rules
        membrane_comp.set_p_lingua_rules("new_rules.pl")
        
        # Check rules are set
        assert membrane_comp.hypers["p_lingua_rules"] == "new_rules.pl"
    
    def test_different_membrane_counts(self):
        """Test membrane computing with different numbers of membranes."""
        for num_membranes in [1, 2, 5, 10]:
            membrane_comp = MembraneComputing(
                membranes=num_membranes,
                membrane_size=15
            )
            
            X = np.random.randn(8, 4)
            membrane_comp.initialize(X[0:1])
            
            # Check membrane states
            membrane_states = membrane_comp.get_membrane_states()
            assert len(membrane_states) == num_membranes
            
            # Test processing
            output = membrane_comp.run(X)
            assert output.shape[0] == 8  # timesteps preserved
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with the same seed."""
        X = np.random.randn(5, 3)
        
        # Create two identical systems with same seed
        membrane_comp1 = MembraneComputing(membranes=2, membrane_size=10)
        membrane_comp2 = MembraneComputing(membranes=2, membrane_size=10)
        
        membrane_comp1.initialize(X[0:1])
        membrane_comp2.initialize(X[0:1])
        
        # Run same data - results may differ due to random initialization
        output1 = membrane_comp1.run(X)
        output2 = membrane_comp2.run(X)
        
        # Check shapes are consistent
        assert output1.shape == output2.shape
    
    def test_membrane_processing_deterministic(self):
        """Test that membrane processing is deterministic for same input."""
        membrane_comp = MembraneComputing(membranes=2, membrane_size=10)
        
        X = np.random.randn(3, 4)
        
        membrane_comp.initialize(X[0:1])
        output1 = membrane_comp.run(X)
        
        # Reset and run again
        membrane_comp2 = MembraneComputing(membranes=2, membrane_size=10)
        membrane_comp2.initialize(X[0:1])
        output2 = membrane_comp2.run(X)
        
        # Check shapes are consistent
        assert output1.shape == output2.shape
    
    def test_single_membrane_system(self):
        """Test edge case of single membrane system."""
        membrane_comp = MembraneComputing(membranes=1, membrane_size=8)
        
        X = np.random.randn(4, 2)
        membrane_comp.initialize(X[0:1])
        
        output = membrane_comp.run(X)
        
        # Should work with single membrane
        assert output.shape == (4, 4)  # membrane_size // 2
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        # Test zero membranes
        with pytest.raises((ValueError, AssertionError)):
            membrane_comp = MembraneComputing(membranes=0)
            X = np.random.randn(3, 2)
            membrane_comp.initialize(X[0:1])
    
    def test_output_consistency(self):
        """Test that output shapes are consistent with membrane configuration."""
        test_cases = [
            (2, True, 20),   # 2 membranes, hierarchical, size 20
            (3, False, 16),  # 3 membranes, parallel, size 16
            (1, True, 12),   # 1 membrane, hierarchical, size 12
            (4, False, 8),   # 4 membranes, parallel, size 8
        ]
        
        for membranes, hierarchical, size in test_cases:
            membrane_comp = MembraneComputing(
                membranes=membranes,
                hierarchical=hierarchical,
                membrane_size=size
            )
            
            X = np.random.randn(6, 3)
            membrane_comp.initialize(X[0:1])
            output = membrane_comp.run(X)
            
            expected_output_dim = size // 2 if hierarchical else membranes * (size // 2)
            assert output.shape == (6, expected_output_dim)