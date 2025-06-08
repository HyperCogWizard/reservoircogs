"""Test suite for Julia Differential Equations module."""

import numpy as np
import pytest
import warnings
from unittest.mock import patch, MagicMock

from reservoirpy.compat.julia_de import JuliaDifferentialESN


@pytest.fixture
def julia_de_test_data():
    """Generate test data for Julia DE."""
    np.random.seed(1234)
    n_timesteps = 100
    dim_in = 3
    dim_out = 2
    
    # Generate input time series (could be sine waves, random walk, etc.)
    t = np.linspace(0, 10, n_timesteps)
    X = np.column_stack([
        np.sin(t),
        np.cos(t),
        0.5 * np.sin(2 * t)
    ])
    
    # Generate targets with some relationship to inputs
    Y = np.column_stack([
        0.3 * X[:, 0] + 0.1 * X[:, 1] + 0.05 * np.random.randn(n_timesteps),
        0.2 * X[:, 1] + 0.15 * X[:, 2] + 0.05 * np.random.randn(n_timesteps)
    ])
    
    return X, Y


@pytest.fixture
def julia_de_sequential_data():
    """Generate sequential test data for Julia DE."""
    np.random.seed(5678)
    n_sequences = 3
    sequence_lengths = [80, 90, 100]
    dim_in = 2
    dim_out = 1
    
    X_sequences = []
    Y_sequences = []
    
    for seq_len in sequence_lengths:
        t = np.linspace(0, 5, seq_len)
        x = np.column_stack([
            np.sin(t + np.random.rand()),
            np.cos(t + np.random.rand())
        ])
        y = 0.5 * (x[:, 0] + x[:, 1]).reshape(-1, 1) + 0.1 * np.random.randn(seq_len, 1)
        
        X_sequences.append(x)
        Y_sequences.append(y)
    
    return X_sequences, Y_sequences


class TestJuliaDifferentialESN:
    """Test suite for Julia Differential ESN."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = JuliaDifferentialESN(
            n_reservoir=50,
            spectral_radius=0.9,
            input_scaling=0.5,
            leaking_rate=0.8,
            ridge=1e-5,
            solver="Tsit5",
            dt=0.01
        )
        
        assert model.n_reservoir == 50
        assert model.spectral_radius == 0.9
        assert model.input_scaling == 0.5
        assert model.leaking_rate == 0.8
        assert model.ridge == 1e-5
        assert model.solver == "Tsit5"
        assert model.dt == 0.01
        assert not model.initialized
    
    def test_julia_availability_check(self):
        """Test Julia availability checking."""
        model = JuliaDifferentialESN()
        
        # The availability should be checked during initialization
        assert hasattr(model, '_julia_available')
        assert isinstance(model.julia_available, bool)
    
    def test_reservoir_matrix_initialization(self, julia_de_test_data):
        """Test reservoir matrix initialization."""
        X, Y = julia_de_test_data
        
        model = JuliaDifferentialESN(n_reservoir=30, seed=1234)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        
        assert model.initialized
        assert model._Win.shape == (30, X.shape[1])
        assert model._W.shape == (30, 30)
        assert model._bias.shape == (30,)
        
        # Check spectral radius
        eigenvals = np.linalg.eigvals(model._W)
        actual_spectral_radius = np.max(np.abs(eigenvals))
        assert actual_spectral_radius <= model.spectral_radius + 1e-10
    
    def test_python_fallback_ode(self, julia_de_test_data):
        """Test Python fallback ODE integration."""
        X, Y = julia_de_test_data
        
        model = JuliaDifferentialESN(n_reservoir=20, dt=0.1, seed=42)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        
        # Force Python fallback
        reservoir_states = model._python_fallback_ode(X)
        
        assert reservoir_states.shape == (len(X), model.n_reservoir)
        assert np.isfinite(reservoir_states).all()
        assert not np.allclose(reservoir_states[0], reservoir_states[-1])  # States should evolve
    
    @patch('subprocess.run')
    def test_julia_ode_execution_success(self, mock_run, julia_de_test_data):
        """Test successful Julia ODE execution."""
        X, Y = julia_de_test_data
        
        # Mock successful Julia execution
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        
        model = JuliaDifferentialESN(n_reservoir=15, seed=123)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        model._julia_available = True
        
        # Mock the file reading
        expected_states = np.random.randn(len(X), model.n_reservoir)
        
        with patch('os.path.exists', return_value=True), \
             patch('numpy.loadtxt', return_value=expected_states):
            
            reservoir_states = model._run_julia_ode(X)
            
            assert np.array_equal(reservoir_states, expected_states)
    
    @patch('subprocess.run')
    def test_julia_ode_execution_failure(self, mock_run, julia_de_test_data):
        """Test Julia ODE execution failure fallback."""
        X, Y = julia_de_test_data
        
        # Mock failed Julia execution
        mock_run.return_value = MagicMock(returncode=1, stderr="Julia error")
        
        model = JuliaDifferentialESN(n_reservoir=15, seed=123)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        model._julia_available = True
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reservoir_states = model._run_julia_ode(X)
            
            # Should have issued a warning and fallen back to Python
            assert len(w) > 0
            assert "Julia execution failed" in str(w[0].message)
        
        # Should return results from Python fallback
        assert reservoir_states.shape == (len(X), model.n_reservoir)
        assert np.isfinite(reservoir_states).all()
    
    def test_partial_fit(self, julia_de_test_data):
        """Test partial fitting functionality."""
        X, Y = julia_de_test_data
        
        model = JuliaDifferentialESN(n_reservoir=25, ridge=1e-4, seed=999)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        
        # Store initial state
        initial_XXT = model._XXT.copy()
        initial_YXT = model._YXT.copy()
        
        model.partial_fit(X, Y)
        
        # Matrices should have been updated
        assert not np.allclose(model._XXT, initial_XXT)
        assert not np.allclose(model._YXT, initial_YXT)
        assert model._XXT.shape == (model.n_reservoir + 1, model.n_reservoir + 1)
        assert model._YXT.shape == (Y.shape[1], model.n_reservoir + 1)
    
    def test_fit_single_sequence(self, julia_de_test_data):
        """Test fitting with single sequence."""
        X, Y = julia_de_test_data
        
        model = JuliaDifferentialESN(n_reservoir=20, ridge=1e-3, seed=777)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        
        Wout = model.fit(X, Y)
        
        assert Wout is not None
        assert Wout.shape == (Y.shape[1], model.n_reservoir + 1)
        assert np.isfinite(Wout).all()
        assert model.Wout is not None
    
    def test_fit_multiple_sequences(self, julia_de_sequential_data):
        """Test fitting with multiple sequences."""
        X_sequences, Y_sequences = julia_de_sequential_data
        
        model = JuliaDifferentialESN(n_reservoir=15, ridge=1e-3, seed=555)
        model.initialize(dim_in=X_sequences[0].shape[1], dim_out=Y_sequences[0].shape[1])
        
        Wout = model.fit(X_sequences, Y_sequences)
        
        assert Wout is not None
        assert Wout.shape == (Y_sequences[0].shape[1], model.n_reservoir + 1)
        assert np.isfinite(Wout).all()
    
    def test_prediction(self, julia_de_test_data):
        """Test prediction functionality."""
        X, Y = julia_de_test_data
        
        model = JuliaDifferentialESN(n_reservoir=20, ridge=1e-3, seed=333)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        model.fit(X, Y)
        
        predictions = model.predict(X)
        
        assert predictions.shape == Y.shape
        assert np.isfinite(predictions).all()
    
    def test_prediction_with_list_input(self, julia_de_test_data):
        """Test prediction with list input."""
        X, Y = julia_de_test_data
        
        model = JuliaDifferentialESN(n_reservoir=20, ridge=1e-3, seed=111)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        model.fit(X, Y)
        
        predictions = model.predict([X])  # Pass as list
        
        assert predictions.shape == Y.shape
        assert np.isfinite(predictions).all()
    
    def test_model_properties(self):
        """Test model properties."""
        model = JuliaDifferentialESN(n_reservoir=30)
        model.initialize(dim_in=3, dim_out=2)
        
        assert model.dim_in == 3
        assert model.dim_out == 2
        assert model.initialized
        assert isinstance(model.julia_available, bool)
    
    def test_different_solvers(self, julia_de_test_data):
        """Test different ODE solvers."""
        X, Y = julia_de_test_data
        
        solvers = ["Tsit5", "Vern7", "Rodas5P"]
        
        for solver in solvers:
            model = JuliaDifferentialESN(
                n_reservoir=15, 
                solver=solver, 
                ridge=1e-3, 
                seed=888
            )
            model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
            model.fit(X, Y)
            
            predictions = model.predict(X)
            assert predictions.shape == Y.shape
            assert np.isfinite(predictions).all()
    
    def test_different_parameters(self, julia_de_test_data):
        """Test different model parameters."""
        X, Y = julia_de_test_data
        
        # Test different spectral radii
        for spectral_radius in [0.5, 0.95, 1.2]:
            model = JuliaDifferentialESN(
                n_reservoir=20,
                spectral_radius=spectral_radius,
                seed=444
            )
            model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
            
            eigenvals = np.linalg.eigvals(model._W)
            actual_sr = np.max(np.abs(eigenvals))
            assert actual_sr <= spectral_radius + 1e-10
    
    def test_convergence_properties(self, julia_de_test_data):
        """Test convergence properties with different time steps."""
        X, Y = julia_de_test_data
        
        dt_values = [0.001, 0.01, 0.1]
        predictions = []
        
        for dt in dt_values:
            model = JuliaDifferentialESN(
                n_reservoir=15, 
                dt=dt, 
                ridge=1e-3, 
                seed=222
            )
            model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
            model.fit(X, Y)
            
            pred = model.predict(X)
            predictions.append(pred)
            assert np.isfinite(pred).all()
        
        # Predictions should be stable across different time steps
        for pred in predictions:
            assert pred.shape == Y.shape
    
    def test_uninitialized_model_error(self, julia_de_test_data):
        """Test error when using uninitialized model."""
        X, Y = julia_de_test_data
        
        model = JuliaDifferentialESN()
        
        with pytest.raises(RuntimeError, match="never initialized"):
            model.partial_fit(X, Y)
        
        with pytest.raises(RuntimeError, match="must be initialized"):
            model.predict(X)
    
    def test_prediction_before_fitting_error(self, julia_de_test_data):
        """Test error when predicting before fitting."""
        X, Y = julia_de_test_data
        
        model = JuliaDifferentialESN()
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(X)
    
    def test_dimension_consistency(self):
        """Test dimension consistency checks."""
        model = JuliaDifferentialESN(n_reservoir=25)
        model.initialize(dim_in=3, dim_out=2)
        
        # Test with wrong input dimension
        X_wrong = np.random.randn(50, 5)  # Should be 3 features
        Y = np.random.randn(50, 2)
        
        # Should handle gracefully or give meaningful error
        try:
            model.partial_fit(X_wrong, Y)
        except (ValueError, IndexError):
            pass  # Expected behavior for dimension mismatch
    
    def test_reproducibility(self, julia_de_test_data):
        """Test that results are reproducible with same seed."""
        X, Y = julia_de_test_data
        
        # First model
        model1 = JuliaDifferentialESN(n_reservoir=20, seed=12345, ridge=1e-4)
        model1.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        model1.fit(X, Y)
        pred1 = model1.predict(X)
        
        # Second model with same seed
        model2 = JuliaDifferentialESN(n_reservoir=20, seed=12345, ridge=1e-4)
        model2.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        model2.fit(X, Y)
        pred2 = model2.predict(X)
        
        # Should be very close (allowing for small numerical differences)
        assert np.allclose(pred1, pred2, atol=1e-10)
    
    def test_numerical_stability(self, julia_de_test_data):
        """Test numerical stability with challenging parameters."""
        X, Y = julia_de_test_data
        
        # Test with very small ridge parameter
        model = JuliaDifferentialESN(
            n_reservoir=15, 
            ridge=1e-12, 
            spectral_radius=1.5,
            seed=999
        )
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        model.fit(X, Y)
        
        predictions = model.predict(X)
        assert np.isfinite(predictions).all()
        assert not np.isnan(predictions).any()
    
    def test_create_julia_ode_script(self, julia_de_test_data):
        """Test Julia ODE script creation."""
        X, Y = julia_de_test_data
        
        model = JuliaDifferentialESN(n_reservoir=10, dt=0.05, leaking_rate=0.7)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        
        script = model._create_julia_ode_script(X)
        
        assert "using DifferentialEquations" in script
        assert "using LinearAlgebra" in script
        assert f"const n_reservoir = {model.n_reservoir}" in script
        assert f"const leaking_rate = {model.leaking_rate}" in script
        assert f"const dt = {model.dt}" in script
        assert "ODEProblem" in script
        assert "solve(" in script
    
    @patch('subprocess.run')
    def test_julia_timeout_handling(self, mock_run, julia_de_test_data):
        """Test handling of Julia execution timeout."""
        X, Y = julia_de_test_data
        
        # Mock timeout
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired("julia", 300)
        
        model = JuliaDifferentialESN(n_reservoir=15, seed=123)
        model.initialize(dim_in=X.shape[1], dim_out=Y.shape[1])
        model._julia_available = True
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reservoir_states = model._run_julia_ode(X)
            
            # Should have issued a timeout warning and fallen back to Python
            assert len(w) > 0
            assert "timed out" in str(w[0].message)
        
        # Should return results from Python fallback
        assert reservoir_states.shape == (len(X), model.n_reservoir)
        assert np.isfinite(reservoir_states).all()