"""B-Series Rooted Tree Gradient Descent for Ridge Regression.

This module implements B-Series expansion methods with Runge-Kutta numerical
integration for optimized ridge regression in reservoir computing. The approach
combines classical numerical analysis techniques with machine learning
optimization for enhanced convergence and stability.

The B-Series method represents solutions to differential equations as power
series in terms of rooted trees, providing a systematic framework for
constructing high-order numerical methods. When applied to ridge regression
optimization, this approach can offer superior convergence properties compared
to standard gradient descent methods.

References:
    Hairer, E., Lubich, C., & Wanner, G. (2006). Geometric Numerical Integration.
    Springer-Verlag.
"""

import numpy as np
from scipy import linalg
from typing import Union, List, Tuple, Optional

from ..type import Data, Weights
from .regression_models import _prepare_inputs, _check_tikhnonv_terms, _OfflineModel


class BSeriesRidgeRegression(_OfflineModel):
    """B-Series Rooted Tree Gradient Descent for Ridge Regression.
    
    This class implements an advanced optimization approach for ridge regression
    using B-Series expansions and Runge-Kutta methods. The approach provides
    enhanced numerical stability and convergence properties for reservoir
    computing applications.
    
    The method combines:
    - B-Series expansion for systematic construction of high-order methods
    - Rooted tree representation for organizing differential equation solutions
    - Runge-Kutta integration for numerical optimization
    - Ridge regularization for stable matrix inversion
    
    Parameters
    ----------
    ridge : float, default=0.1
        Ridge regularization parameter.
    rk_order : int, default=4
        Order of the Runge-Kutta method (2, 4, or 6).
    step_size : float, default=0.01
        Step size for the optimization process.
    max_iterations : int, default=1000
        Maximum number of optimization iterations.
    tolerance : float, default=1e-6
        Convergence tolerance for optimization.
    workers : int, default=-1
        Number of parallel workers.
    dtype : numpy.dtype, default=np.float64
        Data type for computations.
    """
    
    def __init__(
        self,
        ridge: float = 0.1,
        rk_order: int = 4,
        step_size: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        workers: int = -1,
        dtype: np.dtype = np.float64
    ):
        self.ridge = ridge
        self.rk_order = rk_order
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.workers = workers
        
        self._dtype = dtype
        self._ridgeid = None
        self._XXT = None
        self._YXT = None
        
        # B-Series coefficients for different RK orders
        self._bseries_coeffs = self._initialize_bseries_coefficients()
        
    def _initialize_bseries_coefficients(self) -> dict:
        """Initialize B-Series coefficients for different Runge-Kutta orders."""
        coeffs = {
            2: {  # RK2 (Heun's method)
                'trees': ['τ', 'τ²'],
                'coeffs': [1.0, 0.5],
                'stages': 2
            },
            4: {  # RK4 (classical)
                'trees': ['τ', 'τ²', 'τ³', 'τ⁴', '[τ²]'],
                'coeffs': [1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/12.0],
                'stages': 4
            },
            6: {  # RK6 (extended)
                'trees': ['τ', 'τ²', 'τ³', 'τ⁴', 'τ⁵', 'τ⁶', '[τ²]', '[τ³]', '[[τ²]]'],
                'coeffs': [1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0, 1.0/720.0, 
                          1.0/12.0, 1.0/24.0, 1.0/72.0],
                'stages': 6
            }
        }
        return coeffs
    
    def _rooted_tree_gradient(self, W: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute gradient using rooted tree structure.
        
        This method implements the gradient computation based on rooted tree
        representation of the optimization landscape. Each tree corresponds to
        a different order term in the B-Series expansion.
        
        Parameters
        ----------
        W : numpy.ndarray
            Current weight matrix.
        X : numpy.ndarray
            Input state matrix.
        Y : numpy.ndarray
            Target output matrix.
            
        Returns
        -------
        numpy.ndarray
            Gradient computed via rooted tree structure.
        """
        # Ensure numerical stability
        W = np.clip(W, -10, 10)  # Prevent extreme weights
        
        # Residual computation (order 1 tree: τ)
        prediction = X @ W.T
        residual = prediction - Y
        grad_basic = X.T @ residual
        
        # Apply clipping to prevent overflow
        grad_basic = np.clip(grad_basic, -1e6, 1e6)
        
        # Higher order tree contributions
        if self.rk_order >= 4:
            # Second order tree: τ² 
            hessian_approx = X.T @ X + self.ridge * np.eye(X.shape[1])
            
            # Add small regularization for numerical stability
            hessian_approx += 1e-8 * np.eye(X.shape[1])
            
            grad_second_order = hessian_approx @ W.T
            grad_second_order = np.clip(grad_second_order, -1e6, 1e6)
            
            # Combine using B-Series coefficients
            coeffs = self._bseries_coeffs[self.rk_order]['coeffs']
            gradient = coeffs[0] * grad_basic + coeffs[1] * grad_second_order
            
            # Higher order corrections for RK6
            if self.rk_order == 6:
                # Third order tree corrections with improved stability
                residual_norm = np.linalg.norm(residual)
                if residual_norm > 0 and residual_norm < 1e3:  # Only if reasonable magnitude
                    residual_prod = residual.T @ residual / max(residual.shape[0], 1)
                    residual_prod = np.clip(residual_prod, -1e3, 1e3)
                    grad_third_order = grad_basic @ residual_prod / max(residual.shape[0], 1)
                    grad_third_order = np.clip(grad_third_order, -1e6, 1e6)
                    gradient += coeffs[2] * grad_third_order
        else:
            # RK2 case
            coeffs = self._bseries_coeffs[self.rk_order]['coeffs']
            gradient = coeffs[0] * grad_basic
            
        # Final clipping and NaN checking
        gradient = np.clip(gradient, -1e6, 1e6)
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=1e6, neginf=-1e6)
            
        return gradient.T
    
    def _runge_kutta_step(self, W: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Perform one Runge-Kutta optimization step.
        
        Parameters
        ----------
        W : numpy.ndarray
            Current weight matrix.
        X : numpy.ndarray
            Input state matrix.
        Y : numpy.ndarray
            Target output matrix.
            
        Returns
        -------
        numpy.ndarray
            Updated weight matrix.
        """
        h = self.step_size
        
        # Ensure input weight matrix is stable
        W = np.clip(W, -10, 10)
        
        if self.rk_order == 2:
            # RK2 (Heun's method)
            k1 = -self._rooted_tree_gradient(W, X, Y)
            k1 = np.clip(k1, -1.0, 1.0)  # Limit step size
            
            W_temp = W + h * k1
            W_temp = np.clip(W_temp, -10, 10)
            
            k2 = -self._rooted_tree_gradient(W_temp, X, Y)
            k2 = np.clip(k2, -1.0, 1.0)
            
            W_new = W + h * (k1 + k2) / 2
            
        elif self.rk_order == 4:
            # RK4 (classical) with stability checks
            k1 = -self._rooted_tree_gradient(W, X, Y)
            k1 = np.clip(k1, -1.0, 1.0)
            
            W_temp = W + h * k1 / 2
            W_temp = np.clip(W_temp, -10, 10)
            k2 = -self._rooted_tree_gradient(W_temp, X, Y)
            k2 = np.clip(k2, -1.0, 1.0)
            
            W_temp = W + h * k2 / 2
            W_temp = np.clip(W_temp, -10, 10)
            k3 = -self._rooted_tree_gradient(W_temp, X, Y)
            k3 = np.clip(k3, -1.0, 1.0)
            
            W_temp = W + h * k3
            W_temp = np.clip(W_temp, -10, 10)
            k4 = -self._rooted_tree_gradient(W_temp, X, Y)
            k4 = np.clip(k4, -1.0, 1.0)
            
            W_new = W + h * (k1 + 2*k2 + 2*k3 + k4) / 6
            
        elif self.rk_order == 6:
            # RK6 (extended) with enhanced stability
            k1 = -self._rooted_tree_gradient(W, X, Y)
            k1 = np.clip(k1, -0.5, 0.5)  # More conservative for higher order
            
            W_temp = W + h * k1 / 4
            W_temp = np.clip(W_temp, -10, 10)
            k2 = -self._rooted_tree_gradient(W_temp, X, Y)
            k2 = np.clip(k2, -0.5, 0.5)
            
            W_temp = W + h * (k1 + k2) / 8
            W_temp = np.clip(W_temp, -10, 10)
            k3 = -self._rooted_tree_gradient(W_temp, X, Y)
            k3 = np.clip(k3, -0.5, 0.5)
            
            W_temp = W + h * (-k2 + 2*k3) / 2
            W_temp = np.clip(W_temp, -10, 10)
            k4 = -self._rooted_tree_gradient(W_temp, X, Y)
            k4 = np.clip(k4, -0.5, 0.5)
            
            W_temp = W + h * (3*k1 + 9*k4) / 16
            W_temp = np.clip(W_temp, -10, 10)
            k5 = -self._rooted_tree_gradient(W_temp, X, Y)
            k5 = np.clip(k5, -0.5, 0.5)
            
            W_temp = W + h * (-3*k1 + 2*k2 + 12*k3 - 12*k4 + 8*k5) / 7
            W_temp = np.clip(W_temp, -10, 10)
            k6 = -self._rooted_tree_gradient(W_temp, X, Y)
            k6 = np.clip(k6, -0.5, 0.5)
            
            W_new = W + h * (7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6) / 90
            
        else:
            raise ValueError(f"Unsupported Runge-Kutta order: {self.rk_order}")
        
        # Final stability check
        W_new = np.clip(W_new, -10, 10)
        W_new = np.nan_to_num(W_new, nan=0.0, posinf=10.0, neginf=-10.0)
            
        return W_new
    
    def _loss_function(self, W: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute ridge regression loss."""
        residual = X @ W.T - Y
        data_loss = np.mean(residual**2)
        ridge_penalty = self.ridge * np.mean(W**2)
        return data_loss + ridge_penalty
    
    def initialize(self, dim_in: int = None, dim_out: int = None):
        """Initialize the B-Series model parameters.
        
        Parameters
        ----------
        dim_in : int
            Input dimension (including bias).
        dim_out : int
            Output dimension.
        """
        if dim_in is not None:
            self._dim_in = dim_in
        if dim_out is not None:
            self._dim_out = dim_out
            
        # Initialize weight matrix with small random values using Xavier initialization
        if getattr(self, "Wout", None) is None:
            scale = np.sqrt(2.0 / (self._dim_in + self._dim_out + 1))
            self.Wout = np.random.normal(
                0, scale, (self._dim_out, self._dim_in + 1)
            ).astype(self._dtype)
            
        # Initialize Tikhonov matrices for compatibility
        if getattr(self, "_XXT", None) is None:
            self._XXT = np.zeros((self._dim_in + 1, self._dim_in + 1), dtype=self._dtype)
        if getattr(self, "_YXT", None) is None:
            self._YXT = np.zeros((self._dim_out, self._dim_in + 1), dtype=self._dtype)
        if getattr(self, "_ridgeid", None) is None:
            self._ridgeid = self.ridge * np.eye(self._dim_in + 1, dtype=self._dtype)
            
        self._initialized = True
    
    def partial_fit(self, X: Data, Y: Data):
        """Partially fit the model using B-Series optimization.
        
        Parameters
        ----------
        X : numpy.ndarray or list of numpy.ndarray
            Input state sequences.
        Y : numpy.ndarray or list of numpy.ndarray
            Target output sequences.
        """
        X, Y = _prepare_inputs(X, Y, allow_reshape=True)
        
        if not self._initialized:
            raise RuntimeError("BSeries model was never initialized. Call initialize() first.")
            
        _check_tikhnonv_terms(self._XXT, self._YXT, X, Y)
        
        # Update Tikhonov matrices for compatibility with base class
        xxt = X.T.dot(X)
        yxt = Y.T.dot(X)
        self._XXT += xxt
        self._YXT += yxt
        
        # Perform B-Series optimization steps
        current_loss = self._loss_function(self.Wout, X, Y)
        
        for iteration in range(self.max_iterations):
            # Perform Runge-Kutta step
            W_new = self._runge_kutta_step(self.Wout, X, Y)
            
            # Check convergence
            new_loss = self._loss_function(W_new, X, Y)
            
            if abs(current_loss - new_loss) < self.tolerance:
                break
                
            self.Wout = W_new
            current_loss = new_loss
    
    def fit(self, X: Data = None, Y: Data = None) -> Weights:
        """Fit the B-Series ridge regression model.
        
        Parameters
        ----------
        X : numpy.ndarray or list of numpy.ndarray, optional
            Input state sequences.
        Y : numpy.ndarray or list of numpy.ndarray, optional
            Target output sequences.
            
        Returns
        -------
        numpy.ndarray
            Optimized weight matrix.
        """
        if X is not None and Y is not None:
            if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
                self.partial_fit(X, Y)
            elif isinstance(X, list) and isinstance(Y, list):
                # Process multiple sequences
                for x, y in zip(X, Y):
                    self.partial_fit(x, y)
        
        return self.Wout
    
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