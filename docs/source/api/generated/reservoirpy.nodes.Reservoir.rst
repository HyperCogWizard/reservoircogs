reservoirpy.nodes.Reservoir
===========================

.. currentmodule:: reservoirpy.nodes

.. autoclass:: Reservoir
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Pool of leaky-integrator neurons with random recurrent connexions.

Reservoir neurons states, gathered in a vector :math:`x`, may follow one of the two update rules below:

- **1.** Activation function is part of the neuron internal state (equation called ``internal``):

.. math::

   x[t+1]=(1−lr)∗x[t]+lr∗f(Win⋅(u[t+1]+cin∗ξ)+W⋅x[t]+Wfb⋅(g(y[t])+cfb∗ξ)+b)+c∗ξ

- **2.** Activation function is applied on emitted internal states (equation called ``external``):

.. math::

   r[t+1]=(1−lr)∗r[t]+lr∗(Win⋅(u[t+1]+cin∗ξ)+W⋅x[t]+Wfb⋅(g(y[t])+cfb∗ξ)+b)

.. math::

   x[t+1]=f(r[t+1])+c∗ξ

where:

- :math:`x` is the output activation vector of the reservoir;
- :math:`r` is the (optional) internal activation vector of the reservoir;
- :math:`u` is the input timeseries;
- :math:`y` is a feedback vector;
- :math:`ξ` is a random noise;
- :math:`f` and :math:`g` are activation functions.

**Parameters list:**

================== ===================================================================
``W``              Recurrent weights matrix (:math:`W`).
``Win``            Input weights matrix (:math:`Win`).
``Wfb``            Feedback weights matrix (:math:`Wfb`).
``bias``           Input bias vector (:math:`b`).
``internal_state`` Internal state used with equation="external" (:math:`r`).
================== ===================================================================

**Hyperparameters list:**

======================= ========================================================
``lr``                  Leaking rate (1.0 by default) (:math:`lr`).
``sr``                  Spectral radius of ``W`` (optional).
``input_scaling``       Input scaling (float or array) (1.0 by default).
``fb_scaling``          Feedback scaling (float or array) (1.0 by default).
``rc_connectivity``     Connectivity (or density) of ``W`` (0.1 by default).
``input_connectivity``  Connectivity (or density) of ``Win`` (0.1 by default).
``fb_connectivity``     Connectivity (or density) of ``Wfb`` (0.1 by default).
``noise_in``            Input noise gain (0 by default) (:math:`cin∗ξ`).
``noise_rc``            Reservoir state noise gain (0 by default) (:math:`c∗ξ`).
``noise_fb``            Feedback noise gain (0 by default) (:math:`cfb∗ξ`).
``noise_type``          Distribution of noise (normal by default) (:math:`ξ∼Noise type`).
``activation``          Activation of the reservoir units (tanh by default) (:math:`f`).
``fb_activation``       Activation of the feedback units (identity by default) (:math:`g`).
``units``               Number of neuronal units in the reservoir.
``noise_generator``     A random state generator.
======================= ========================================================

Parameters
----------

units : int, optional
    Number of reservoir units. If None, the number of units will be inferred from the ``W`` matrix shape.
lr : float or array-like of shape (units,), default to 1.0
    Neurons leak rate. Must be in :math:`[0,1]`.
sr : float, optional
    Spectral radius of recurrent weight matrix.
input_bias : bool, default to True
    If False, no bias is added to inputs.
noise_rc : float, default to 0.0
    Gain of noise applied to reservoir activations.
noise_in : float, default to 0.0
    Gain of noise applied to input inputs.
noise_fb : float, default to 0.0
    Gain of noise applied to feedback signal.
noise_type : str, default to "normal"
    Distribution of noise. Must be a Numpy random variable generator distribution (see `numpy.random.Generator`).
noise_kwargs : dict, optional
    Keyword arguments to pass to the noise generator, such as low and high values of uniform distribution.
input_scaling : float or array-like of shape (features,), default to 1.0.
    Input gain. An array of the same dimension as the inputs can be used to set up different input scaling for each feature.
bias_scaling : float, default to 1.0
    Bias gain.
fb_scaling : float or array-like of shape (features,), default to 1.0
    Feedback gain. An array of the same dimension as the feedback can be used to set up different feedback scaling for each feature.
input_connectivity : float, default to 0.1
    Connectivity of input neurons, i.e. ratio of input neurons connected to reservoir neurons. Must be in :math:`]0,1]`.
rc_connectivity : float, default to 0.1
    Connectivity of recurrent weight matrix, i.e. ratio of reservoir neurons connected to other reservoir neurons, including themselves. Must be in :math:`]0,1]`.
fb_connectivity : float, default to 0.1
    Connectivity of feedback neurons, i.e. ratio of feedback neurons connected to reservoir neurons. Must be in :math:`]0,1]`.
Win : callable or array-like of shape (units, features), default to :func:`~reservoirpy.mat_gen.bernoulli`
    Input weights matrix or initializer. If a callable (like a function) is used, then this function should accept any keywords parameters and at least two parameters that will be used to define the shape of the returned weight matrix.
W : callable or array-like of shape (units, units), default to :func:`~reservoirpy.mat_gen.normal`
    Recurrent weights matrix or initializer. If a callable (like a function) is used, then this function should accept any keywords parameters and at least two parameters that will be used to define the shape of the returned weight matrix.
bias : callable or array-like of shape (units, 1), default to :func:`~reservoirpy.mat_gen.bernoulli`
    Bias weights vector or initializer. If a callable (like a function) is used, then this function should accept any keywords parameters and at least two parameters that will be used to define the shape of the returned weight matrix.
Wfb : callable or array-like of shape (units, feedback), default to :func:`~reservoirpy.mat_gen.bernoulli`
    Feedback weights matrix or initializer. If a callable (like a function) is used, then this function should accept any keywords parameters and at least two parameters that will be used to define the shape of the returned weight matrix.
fb_activation : str or callable, default to :func:`~reservoirpy.activationsfunc.identity`
    Feedback activation function. - If a str, should be a :mod:`~reservoirpy.activationsfunc` function name. - If a callable, should be an element-wise operator on arrays.
activation : str or callable, default to :func:`~reservoirpy.activationsfunc.tanh`
    Reservoir units activation function. - If a str, should be a :mod:`~reservoirpy.activationsfunc` function name. - If a callable, should be an element-wise operator on arrays.
equation : {"internal", "external"}, default to "internal"
    If "internal", will use equation defined in equation 1 to update the state of reservoir units. If "external", will use the equation defined in equation 2 (see above).
feedback_dim : int, optional
    Feedback dimension. Can be inferred at first call.
input_dim : int, optional
    Input dimension. Can be inferred at first call.
name : str, optional
    Node name.
dtype : Numpy dtype, default to np.float64
    Numerical type for node parameters.
seed : int or `numpy.random.Generator`, optional
    A random state seed, for noise generation.

Note
----

If W, Win, bias or Wfb are initialized with an array-like matrix, then all initializers parameters such as spectral radius (``sr``) or input scaling (``input_scaling``) are ignored. See :mod:`~reservoirpy.mat_gen` for more information.

Example
-------

>>> from reservoirpy.nodes import Reservoir
>>> reservoir = Reservoir(100, lr=0.2, sr=0.8) # a 100 neurons reservoir

Using the :func:`~reservoirpy.datasets.mackey_glass` timeseries:

>>> from reservoirpy.datasets import mackey_glass
>>> x = mackey_glass(200)
>>> states = reservoir.run(x)

Methods
-------

.. autosummary::
   :toctree: ./

   ~Reservoir.__init__
   ~Reservoir.call
   ~Reservoir.clean_buffers
   ~Reservoir.copy
   ~Reservoir.create_buffer
   ~Reservoir.feedback
   ~Reservoir.fit
   ~Reservoir.get_buffer
   ~Reservoir.get_param
   ~Reservoir.initialize
   ~Reservoir.initialize_buffers
   ~Reservoir.initialize_feedback
   ~Reservoir.link_feedback
   ~Reservoir.partial_fit
   ~Reservoir.reset
   ~Reservoir.run
   ~Reservoir.set_buffer
   ~Reservoir.set_feedback_dim
   ~Reservoir.set_input_dim
   ~Reservoir.set_output_dim
   ~Reservoir.set_param
   ~Reservoir.set_state_proxy
   ~Reservoir.state
   ~Reservoir.state_proxy
   ~Reservoir.train
   ~Reservoir.with_feedback
   ~Reservoir.with_state
   ~Reservoir.zero_feedback
   ~Reservoir.zero_state

Attributes
----------

.. autosummary::

   ~Reservoir.dtype
   ~Reservoir.feedback_dim
   ~Reservoir.fitted
   ~Reservoir.has_feedback
   ~Reservoir.hypers
   ~Reservoir.input_dim
   ~Reservoir.is_fb_initialized
   ~Reservoir.is_initialized
   ~Reservoir.is_trainable
   ~Reservoir.is_trained_offline
   ~Reservoir.is_trained_online
   ~Reservoir.name
   ~Reservoir.output_dim
   ~Reservoir.params
   ~Reservoir.unsupervised