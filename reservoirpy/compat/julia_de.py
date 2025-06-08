"""J-Surface Julia Differential Equations for Reservoir Computing.

This module implements Julia-based differential equation solving with elementary
differential echo state networks. It leverages DifferentialEquations.jl and
ModelingToolkit.jl for high-performance numerical integration within reservoir
computing frameworks.

The J-Surface method provides:
- Elementary differential echo state networks (ESNs)
- Julia-based ODE/SDE solving with DifferentialEquations.jl
- Symbolic modeling capabilities via ModelingToolkit.jl
- Integration with reservoir computing optimization

References:
    Rackauckas, C., & Nie, Q. (2017). DifferentialEquations.jl–a performant
    and feature-rich ecosystem for solving differential equations in Julia.
    Journal of Open Research Software, 5(1).
"""

import numpy as np
import warnings
from typing import Union, List, Tuple, Optional, Callable
import subprocess
import tempfile
import os

from ..type import Data, Weights
from .regression_models import _prepare_inputs, _check_tikhnonv_terms, _OfflineModel


class JuliaDifferentialESN(_OfflineModel):
    """Julia-based Differential Echo State Network.
    
    This class implements an elementary differential echo state network using
    Julia's DifferentialEquations.jl for numerical integration. The approach
    provides enhanced modeling capabilities for time-series prediction and
    dynamic systems modeling.
    
    Parameters
    ----------
    n_reservoir : int, default=100
        Number of neurons in the reservoir.
    spectral_radius : float, default=0.95
        Spectral radius of the reservoir weight matrix.
    input_scaling : float, default=1.0
        Scaling factor for input weights.
    leaking_rate : float, default=1.0
        Leaking rate (alpha) of the ESN.
    ridge : float, default=1e-6
        Ridge regularization parameter for readout training.
    solver : str, default="Tsit5"
        Julia ODE solver to use (e.g., "Tsit5", "Vern7", "Rodas5P").
    dt : float, default=0.01
        Time step for differential equation integration.
    abstol : float, default=1e-6
        Absolute tolerance for ODE solver.
    reltol : float, default=1e-3
        Relative tolerance for ODE solver.
    seed : int, default=None
        Random seed for reproducibility.
    workers : int, default=-1
        Number of parallel workers.
    dtype : numpy.dtype, default=np.float64
        Data type for computations.
    """
    
    def __init__(
        self,
        n_reservoir: int = 100,
        spectral_radius: float = 0.95,
        input_scaling: float = 1.0,
        leaking_rate: float = 1.0,
        ridge: float = 1e-6,
        solver: str = "Tsit5",
        dt: float = 0.01,
        abstol: float = 1e-6,
        reltol: float = 1e-3,
        seed: int = None,
        workers: int = -1,
        dtype: np.dtype = np.float64
    ):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.ridge = ridge
        self.solver = solver
        self.dt = dt
        self.abstol = abstol
        self.reltol = reltol
        self.seed = seed
        self.workers = workers
        
        self._dtype = dtype
        self._initialized = False
        self._julia_available = self._check_julia_availability()
        
        # Initialize reservoir matrices
        self._Win = None
        self._W = None
        self._bias = None
        
        # Tikhonov matrices for compatibility with base class
        self._XXT = None
        self._YXT = None
        self._ridgeid = None
        
    def _check_julia_availability(self) -> bool:
        """Check if Julia and required packages are available."""
        try:
            # Try to run a simple Julia command
            result = subprocess.run(
                ["julia", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                warnings.warn("Julia is not available. Falling back to Python implementation.")
                return False
                
            # Check for DifferentialEquations.jl
            result = subprocess.run([
                "julia", "-e", 
                "using DifferentialEquations; println(\"DifferentialEquations.jl available\")"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                warnings.warn("DifferentialEquations.jl not available. Falling back to Python implementation.")
                return False
                
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            warnings.warn("Julia setup failed. Using Python fallback implementation.")
            return False
    
    def _initialize_reservoir_matrices(self):
        """Initialize reservoir weight matrices."""
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Input weights
        self._Win = np.random.uniform(
            -self.input_scaling, 
            self.input_scaling,
            (self.n_reservoir, self._dim_in)
        ).astype(self._dtype)
        
        # Reservoir weights
        self._W = np.random.randn(self.n_reservoir, self.n_reservoir).astype(self._dtype)
        
        # Scale to desired spectral radius
        current_spectral_radius = np.max(np.abs(np.linalg.eigvals(self._W)))
        if current_spectral_radius > 0:
            self._W *= self.spectral_radius / current_spectral_radius
            
        # Bias
        self._bias = np.random.uniform(-0.1, 0.1, self.n_reservoir).astype(self._dtype)
    
    def _create_julia_ode_script(self, X: np.ndarray) -> str:
        """Create Julia script for ODE solving."""
        script_content = f"""
using DifferentialEquations
using LinearAlgebra

# Reservoir parameters
const n_reservoir = {self.n_reservoir}
const leaking_rate = {self.leaking_rate}
const dt = {self.dt}

# Load weight matrices
Win = reshape([{','.join(map(str, self._Win.flatten()))}], {self._Win.shape[0]}, {self._Win.shape[1]})
W = reshape([{','.join(map(str, self._W.flatten()))}], {self._W.shape[0]}, {self._W.shape[1]})
bias = [{','.join(map(str, self._bias))}]

# Input data
input_data = reshape([{','.join(map(str, X.flatten()))}], {X.shape[0]}, {X.shape[1]})

# ODE function for reservoir dynamics
function reservoir_ode!(du, u, p, t)
    step_idx = min(Int(floor(t / dt)) + 1, size(input_data, 1))
    current_input = input_data[step_idx, :]
    
    # Reservoir dynamics: du/dt = -α*u + α*tanh(Win*input + W*u + bias)
    activation = tanh.(Win * current_input + W * u + bias)
    du .= -leaking_rate * u + leaking_rate * activation
end

# Initial condition
u0 = zeros({self.n_reservoir})

# Time span
tspan = (0.0, {(len(X) - 1) * self.dt})

# Solve ODE
prob = ODEProblem(reservoir_ode!, u0, tspan)
sol = solve(prob, {self.solver}(), 
           abstol={self.abstol}, 
           reltol={self.reltol},
           saveat=dt)

# Extract solution at time points
reservoir_states = reduce(hcat, sol.u)'

# Save to file
using DelimitedFiles
writedlm("reservoir_states.csv", reservoir_states, ',')
"""
        return script_content
    
    def _run_julia_ode(self, X: np.ndarray) -> np.ndarray:
        """Run Julia ODE solver and return reservoir states."""
        if not self._julia_available:
            return self._python_fallback_ode(X)
            
        # Create temporary directory for Julia execution
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "reservoir_ode.jl")
            output_path = os.path.join(temp_dir, "reservoir_states.csv")
            
            # Write Julia script
            script_content = self._create_julia_ode_script(X)
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            try:
                # Run Julia script
                result = subprocess.run(
                    ["julia", script_path],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode != 0:
                    warnings.warn(f"Julia execution failed: {result.stderr}. Using Python fallback.")
                    return self._python_fallback_ode(X)
                
                # Read results
                if os.path.exists(output_path):
                    reservoir_states = np.loadtxt(output_path, delimiter=',')
                    return reservoir_states
                else:
                    warnings.warn("Julia output file not found. Using Python fallback.")
                    return self._python_fallback_ode(X)
                    
            except subprocess.TimeoutExpired:
                warnings.warn("Julia execution timed out. Using Python fallback.")
                return self._python_fallback_ode(X)
            except Exception as e:
                warnings.warn(f"Julia execution error: {e}. Using Python fallback.")
                return self._python_fallback_ode(X)
    
    def _python_fallback_ode(self, X: np.ndarray) -> np.ndarray:
        """Python fallback implementation using Euler integration."""
        n_steps = len(X)
        reservoir_states = np.zeros((n_steps, self.n_reservoir), dtype=self._dtype)
        
        # Initial state
        state = np.zeros(self.n_reservoir, dtype=self._dtype)
        
        for t in range(n_steps):
            # Current input
            current_input = X[t]
            
            # Reservoir dynamics using Euler integration
            activation = np.tanh(self._Win @ current_input + self._W @ state + self._bias)
            
            # du/dt = -α*u + α*activation
            dstate_dt = -self.leaking_rate * state + self.leaking_rate * activation
            
            # Euler step
            state = state + self.dt * dstate_dt
            reservoir_states[t] = state
            
        return reservoir_states
    
    def initialize(self, dim_in: int = None, dim_out: int = None):
        """Initialize the differential ESN model.
        
        Parameters
        ----------
        dim_in : int
            Input dimension.
        dim_out : int
            Output dimension.
        """
        if dim_in is not None:
            self._dim_in = dim_in
        if dim_out is not None:
            self._dim_out = dim_out
            
        # Initialize reservoir matrices
        self._initialize_reservoir_matrices()
        
        # Initialize readout weights (but don't set Wout yet, wait for fitting)
        if getattr(self, "Wout", None) is None:
            # Will be set during fitting
            self.Wout = None
            
        # Initialize Tikhonov matrices for compatibility
        readout_dim = self.n_reservoir + 1
        if getattr(self, "_XXT", None) is None:
            self._XXT = np.zeros((readout_dim, readout_dim), dtype=self._dtype)
        if getattr(self, "_YXT", None) is None:
            self._YXT = np.zeros((self._dim_out, readout_dim), dtype=self._dtype)
        if getattr(self, "_ridgeid", None) is None:
            self._ridgeid = self.ridge * np.eye(readout_dim, dtype=self._dtype)
            
        self._initialized = True
    
    def partial_fit(self, X: Data, Y: Data):
        """Partially fit the model using differential ESN.
        
        Parameters
        ----------
        X : numpy.ndarray or list of numpy.ndarray
            Input sequences.
        Y : numpy.ndarray or list of numpy.ndarray
            Target output sequences.
        """
        X, Y = _prepare_inputs(X, Y, bias=False, allow_reshape=True)
        
        if not self._initialized:
            raise RuntimeError("JuliaDifferentialESN model was never initialized. Call initialize() first.")
            
        # Run differential equation to get reservoir states
        reservoir_states = self._run_julia_ode(X)
        
        # Add bias term to reservoir states
        bias_states = np.ones((reservoir_states.shape[0], 1), dtype=self._dtype)
        extended_states = np.hstack([reservoir_states, bias_states])
        
        # Check and update Tikhonov matrices
        _check_tikhnonv_terms(self._XXT, self._YXT, extended_states, Y)
        
        # Update accumulation matrices
        xxt = extended_states.T @ extended_states
        yxt = Y.T @ extended_states
        self._XXT += xxt
        self._YXT += yxt
    
    def fit(self, X: Data = None, Y: Data = None) -> Weights:
        """Fit the differential ESN model.
        
        Parameters
        ----------
        X : numpy.ndarray or list of numpy.ndarray, optional
            Input sequences.
        Y : numpy.ndarray or list of numpy.ndarray, optional
            Target output sequences.
            
        Returns
        -------
        numpy.ndarray
            Trained readout weights.
        """
        if X is not None and Y is not None:
            if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
                self.partial_fit(X, Y)
            elif isinstance(X, list) and isinstance(Y, list):
                # Process multiple sequences
                for x, y in zip(X, Y):
                    self.partial_fit(x, y)
        
        # Solve for readout weights using ridge regression
        try:
            self.Wout = np.linalg.solve(
                self._XXT + self._ridgeid, 
                self._YXT.T
            ).T
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            regularized_matrix = self._XXT + self._ridgeid
            self.Wout = np.linalg.pinv(regularized_matrix) @ self._YXT.T
            self.Wout = self.Wout.T
        
        return self.Wout
    
    def predict(self, X: Data) -> np.ndarray:
        """Predict outputs using the trained differential ESN.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input sequence.
            
        Returns
        -------
        numpy.ndarray
            Predicted outputs.
        """
        if not self._initialized:
            raise RuntimeError("Model must be initialized before prediction.")
        if self.Wout is None:
            raise RuntimeError("Model must be fitted before prediction.")
            
        if isinstance(X, list):
            X = X[0]  # Take first sequence if list provided
            
        # Run differential equation to get reservoir states
        reservoir_states = self._run_julia_ode(X)
        
        # Add bias term
        bias_states = np.ones((reservoir_states.shape[0], 1), dtype=self._dtype)
        extended_states = np.hstack([reservoir_states, bias_states])
        
        # Compute predictions
        predictions = extended_states @ self.Wout.T
        
        return predictions
    
    @property
    def dim_in(self):
        """Input dimension of the model."""
        return self._dim_in
    
    @property
    def dim_out(self):
        """Output dimension of the model."""
        return self._dim_out
    
    @property
    def initialized(self):
        """Check if model is initialized."""
        return self._initialized
    
    @property
    def julia_available(self):
        """Check if Julia backend is available."""
        return self._julia_available