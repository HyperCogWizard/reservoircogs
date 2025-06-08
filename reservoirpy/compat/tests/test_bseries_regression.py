"""Tests for B-Series Rooted Tree Gradient Descent implementation."""

import numpy as np
import pytest

from ..bseries_regression import BSeriesRidgeRegression


@pytest.fixture(scope="session")
def bseries_test_data():
    """Generate test data for B-Series regression."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    n_outputs = 3
    
    X = np.random.randn(n_samples, n_features)
    W_true = np.random.randn(n_outputs, n_features + 1)  # Include bias
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    Y = X_with_bias @ W_true.T + 0.1 * np.random.randn(n_samples, n_outputs)
    
    return X, Y


@pytest.fixture(scope="session")
def bseries_sequential_data():
    """Generate sequential test data."""
    np.random.seed(42)
    sequences = []
    targets = []
    
    for _ in range(5):
        n_samples = np.random.randint(50, 100)
        n_features = 10
        n_outputs = 2
        
        X = np.random.randn(n_samples, n_features)
        W_true = np.random.randn(n_outputs, n_features + 1)
        X_with_bias = np.column_stack([np.ones(n_samples), X])
        Y = X_with_bias @ W_true.T + 0.1 * np.random.randn(n_samples, n_outputs)
        
        sequences.append(X)
        targets.append(Y)
    
    return sequences, targets


class TestBSeriesRidgeRegression:
    """Test suite for B-Series Ridge Regression."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = BSeriesRidgeRegression(ridge=0.1, rk_order=4)
        assert model.ridge == 0.1
        assert model.rk_order == 4
        assert not model.initialized
        
        # Initialize with dimensions
        model.initialize(dim_in=10, dim_out=3)
        assert model.initialized
        assert model.dim_in == 10
        assert model.dim_out == 3
        assert model.Wout.shape == (3, 11)  # Include bias
    
    def test_bseries_coefficients(self):
        """Test B-Series coefficient initialization."""
        # Test RK2
        model = BSeriesRidgeRegression(rk_order=2)
        coeffs = model._bseries_coeffs[2]
        assert len(coeffs['trees']) == 2
        assert len(coeffs['coeffs']) == 2
        assert coeffs['stages'] == 2
        
        # Test RK4
        model = BSeriesRidgeRegression(rk_order=4)
        coeffs = model._bseries_coeffs[4]
        assert len(coeffs['trees']) == 5
        assert len(coeffs['coeffs']) == 5
        assert coeffs['stages'] == 4
        
        # Test RK6
        model = BSeriesRidgeRegression(rk_order=6)
        coeffs = model._bseries_coeffs[6]
        assert len(coeffs['trees']) == 9
        assert len(coeffs['coeffs']) == 9
        assert coeffs['stages'] == 6
    
    def test_rooted_tree_gradient(self, bseries_test_data):
        """Test rooted tree gradient computation."""
        X, Y = bseries_test_data
        model = BSeriesRidgeRegression(ridge=0.1, rk_order=4)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        
        # Add bias to X for gradient computation
        X_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        gradient = model._rooted_tree_gradient(model.Wout, X_bias, Y)
        
        assert gradient.shape == model.Wout.shape
        assert np.isfinite(gradient).all()
    
    def test_runge_kutta_step(self, bseries_test_data):
        """Test Runge-Kutta optimization step."""
        X, Y = bseries_test_data
        
        for rk_order in [2, 4, 6]:
            model = BSeriesRidgeRegression(ridge=0.1, rk_order=rk_order, step_size=0.01)
            model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
            
            # Add bias to X
            X_bias = np.column_stack([np.ones(X.shape[0]), X])
            
            W_initial = model.Wout.copy()
            W_updated = model._runge_kutta_step(model.Wout, X_bias, Y)
            
            assert W_updated.shape == W_initial.shape
            assert np.isfinite(W_updated).all()
            # Weights should change
            assert not np.allclose(W_initial, W_updated)
    
    def test_loss_function(self, bseries_test_data):
        """Test loss function computation."""
        X, Y = bseries_test_data
        model = BSeriesRidgeRegression(ridge=0.1)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        
        # Add bias to X
        X_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        loss = model._loss_function(model.Wout, X_bias, Y)
        
        assert isinstance(loss, float)
        assert loss >= 0
        assert np.isfinite(loss)
    
    def test_partial_fit(self, bseries_test_data):
        """Test partial fitting functionality."""
        X, Y = bseries_test_data
        model = BSeriesRidgeRegression(ridge=0.1, max_iterations=10)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        
        W_initial = model.Wout.copy()
        model.partial_fit(X, Y)
        
        # Weights should be updated
        assert not np.allclose(W_initial, model.Wout)
        assert np.isfinite(model.Wout).all()
    
    def test_fit_single_sequence(self, bseries_test_data):
        """Test fitting with single sequence."""
        X, Y = bseries_test_data
        model = BSeriesRidgeRegression(ridge=0.1, max_iterations=20)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        
        Wout = model.fit(X, Y)
        
        assert Wout.shape == (Y.shape[1], X.shape[1] + 1)
        assert np.isfinite(Wout).all()
    
    def test_fit_multiple_sequences(self, bseries_sequential_data):
        """Test fitting with multiple sequences."""
        X_list, Y_list = bseries_sequential_data
        model = BSeriesRidgeRegression(ridge=0.1, max_iterations=5)
        model.initialize(dim_in=X_list[0].shape[1], dim_out=Y_list[0].shape[1])
        
        Wout = model.fit(X_list, Y_list)
        
        assert Wout.shape == (Y_list[0].shape[1], X_list[0].shape[1] + 1)
        assert np.isfinite(Wout).all()
    
    def test_convergence_tolerance(self, bseries_test_data):
        """Test convergence based on tolerance."""
        X, Y = bseries_test_data
        model = BSeriesRidgeRegression(
            ridge=0.1, 
            max_iterations=1000, 
            tolerance=1e-8,
            step_size=0.001
        )
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        
        # Track loss over iterations
        X_bias = np.column_stack([np.ones(X.shape[0]), X])
        initial_loss = model._loss_function(model.Wout, X_bias, Y)
        
        model.partial_fit(X, Y)
        
        final_loss = model._loss_function(model.Wout, X_bias, Y)
        
        # Loss should decrease
        assert final_loss <= initial_loss
    
    def test_different_rk_orders_convergence(self, bseries_test_data):
        """Test that different RK orders all converge."""
        X, Y = bseries_test_data
        X_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        losses = {}
        
        for rk_order in [2, 4, 6]:
            model = BSeriesRidgeRegression(
                ridge=0.1, 
                rk_order=rk_order,
                max_iterations=20,
                step_size=0.01
            )
            model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
            
            initial_loss = model._loss_function(model.Wout, X_bias, Y)
            model.partial_fit(X, Y)
            final_loss = model._loss_function(model.Wout, X_bias, Y)
            
            losses[rk_order] = (initial_loss, final_loss)
            
            # All orders should reduce loss
            assert final_loss <= initial_loss
    
    def test_model_properties(self):
        """Test model properties."""
        model = BSeriesRidgeRegression(ridge=0.2)
        
        # Before initialization
        assert not model.initialized
        
        # After initialization
        model.initialize(dim_in=5, dim_out=2)
        assert model.initialized
        assert model.dim_in == 5
        assert model.dim_out == 2
    
    def test_invalid_rk_order(self):
        """Test invalid Runge-Kutta order."""
        model = BSeriesRidgeRegression(rk_order=8)  # Invalid order
        model.initialize(dim_in=5, dim_out=2)
        
        X = np.random.randn(10, 6)  # 5 features + bias
        Y = np.random.randn(10, 2)
        
        with pytest.raises(ValueError):
            model._runge_kutta_step(model.Wout, X, Y)
    
    def test_uninitialized_model_error(self, bseries_test_data):
        """Test error when using uninitialized model."""
        X, Y = bseries_test_data
        model = BSeriesRidgeRegression()
        
        with pytest.raises(RuntimeError):
            model.partial_fit(X, Y)
    
    def test_dimension_mismatch_error(self):
        """Test dimension mismatch errors."""
        model = BSeriesRidgeRegression()
        model.initialize(dim_in=5, dim_out=2)
        
        # Wrong input dimension
        X_wrong = np.random.randn(10, 4)  # Should be 5
        Y = np.random.randn(10, 2)
        
        with pytest.raises(ValueError):
            model.partial_fit(X_wrong, Y)
        
        # Wrong output dimension
        X = np.random.randn(10, 5)
        Y_wrong = np.random.randn(10, 3)  # Should be 2
        
        with pytest.raises(ValueError):
            model.partial_fit(X, Y_wrong)
    
    def test_reproducibility(self, bseries_test_data):
        """Test that results are reproducible with same initialization."""
        X, Y = bseries_test_data
        
        # Run twice with same random seed for weights
        results = []
        for _ in range(2):
            model = BSeriesRidgeRegression(ridge=0.1, max_iterations=10)
            model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
            
            # Set same initial weights
            np.random.seed(123)
            model.Wout = np.random.normal(0, 0.01, model.Wout.shape)
            
            model.partial_fit(X, Y)
            results.append(model.Wout.copy())
        
        # Results should be identical
        np.testing.assert_allclose(results[0], results[1], rtol=1e-10)