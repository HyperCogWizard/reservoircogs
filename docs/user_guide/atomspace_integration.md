# AtomSpace Integration Guide

## Overview

ReservoirCogs provides deep integration with OpenCog's AtomSpace, enabling symbolic AI capabilities alongside traditional reservoir computing. This guide explains how to use these features.

## Core Concepts

### AtomSpace Types for Reservoir Computing

ReservoirCogs defines custom AtomSpace types specifically for reservoir computing:

```
RESERVOIR_NODE              - Base reservoir computing node
ECHO_STATE_NETWORK_NODE     - Specific ESN implementation
RESERVOIR_MATRIX_NODE       - Weight matrices and connections
RESERVOIR_STATE_NODE        - Dynamic state information
ACTIVATION_FUNCTION_NODE    - Nonlinear activation functions
TRAINING_DATA_NODE          - Training datasets
LEARNING_RULE_NODE          - Learning algorithms
```

### Creating Reservoir Atoms

```cpp
#include <opencog/reservoir/types/reservoir_types.h>

// Create reservoir node in AtomSpace
auto atomspace = createAtomSpace();
Handle reservoir_handle = atomspace->add_node(ECHO_STATE_NETWORK_NODE, "my_esn");

// Store configuration as values
Handle config_key = atomspace->add_node(PREDICATE_NODE, "reservoir_size");
reservoir_handle->setValue(config_key, createFloatValue({100.0}));
```

## C++ API Usage

### Basic Reservoir Creation

```cpp
#include <opencog/reservoir/nodes/ReservoirNode.h>
#include <opencog/reservoir/algorithms/ReservoirAlgorithms.h>

using namespace opencog::reservoir;

// Create AtomSpace and ESN
auto atomspace = createAtomSpace();
auto esn = std::make_shared<EchoStateNetwork>(atomspace, 100, 3, 1);

// Configure reservoir parameters
esn->set_leaking_rate(0.3);
esn->set_spectral_radius(0.9);
esn->set_input_scaling(1.0);
```

### Training with AtomSpace Integration

```cpp
using namespace opencog::reservoir::algorithms;

// Create trainer with AtomSpace support
ReservoirTrainer trainer(atomspace);

// Prepare training data
std::vector<std::vector<double>> train_inputs = {{1.0, 0.5, -0.3}, {0.8, -0.2, 0.7}};
std::vector<std::vector<double>> train_targets = {{0.5}, {-0.1}};

// Train with ridge regression
trainer.train_esn_ridge_regression(esn, train_inputs, train_targets, 1e-6);

// Store training results in AtomSpace
esn->store_weights_to_atomspace();
```

### Performance Analysis

```cpp
// Compute reservoir metrics
double memory_cap = ReservoirMetrics::memory_capacity(esn, 50);
double mse = ReservoirMetrics::mean_squared_error(predictions, targets);

// Store metrics in AtomSpace
std::vector<double> metrics = {memory_cap, mse};
Handle perf_handle = ReservoirMetrics::compute_performance_atoms(atomspace, metrics);
```

### Evolutionary Optimization

```cpp
// Create evolutionary optimizer
ReservoirEvolution evolution(atomspace, 50, 0.1); // pop_size=50, mutation_rate=0.1

// Evolve reservoir topology
auto best_esn = evolution.evolve_topology(train_inputs, train_targets, 100);

// Analyze evolved dynamics
auto lyapunov_exp = ReservoirAnalysis::compute_lyapunov_exponents(best_esn, test_inputs);
Handle dynamics_graph = ReservoirAnalysis::create_dynamics_graph(atomspace, best_esn);
```

## Symbolic Reasoning with Reservoirs

### Pattern Recognition

```cpp
// Create pattern recognition atoms
Handle pattern1 = atomspace->add_node(CONCEPT_NODE, "sequence_pattern_1");
Handle pattern2 = atomspace->add_node(CONCEPT_NODE, "sequence_pattern_2");

// Link reservoir states to symbolic concepts
Handle state_link = atomspace->add_link(EVALUATION_LINK,
    atomspace->add_node(PREDICATE_NODE, "recognizes_pattern"),
    {esn->get_handle(), pattern1}
);

// Set confidence based on reservoir output
state_link->setTruthValue(SimpleTruthValue::createTV(0.8, 0.9));
```

### Temporal Logic Integration

```cpp
// Create temporal sequences
Handle seq_at_t1 = atomspace->add_link(AT_TIME_LINK,
    esn->get_handle(),
    atomspace->add_node(TIME_NODE, "t1")
);

Handle seq_at_t2 = atomspace->add_link(AT_TIME_LINK,
    esn->get_handle(), 
    atomspace->add_node(TIME_NODE, "t2")
);

// Express temporal relationships
Handle temporal_relation = atomspace->add_link(SEQUENTIAL_AND_LINK,
    seq_at_t1, seq_at_t2
);
```

### Knowledge Extraction

```cpp
// Extract learned patterns from reservoir
ReservoirAnalysis::analyze_information_flow(atomspace, esn);

// Query for emergent concepts
Handle query = atomspace->add_link(BIND_LINK,
    atomspace->add_link(VARIABLE_LIST,
        atomspace->add_node(VARIABLE_NODE, "$pattern")
    ),
    atomspace->add_link(AND_LINK,
        atomspace->add_link(EVALUATION_LINK,
            atomspace->add_node(PREDICATE_NODE, "learned_by_reservoir"),
            {esn->get_handle(), atomspace->add_node(VARIABLE_NODE, "$pattern")}
        )
    ),
    atomspace->add_node(VARIABLE_NODE, "$pattern")
);

// Execute query to find learned patterns
auto query_result = query->execute(atomspace);
```

## Python-C++ Bridge

### Using C++ Reservoirs from Python

```python
import ctypes
from reservoirpy.opencog_bridge import AtomSpaceReservoir

# Create reservoir with AtomSpace backend
reservoir = AtomSpaceReservoir(size=100, input_dim=3, output_dim=1)

# Configure using Python API
reservoir.set_leaking_rate(0.3)
reservoir.set_spectral_radius(0.9)

# Train with NumPy arrays
import numpy as np
X_train = np.random.randn(1000, 3)
y_train = np.random.randn(1000, 1)

reservoir.fit(X_train, y_train)
predictions = reservoir.predict(X_test)

# Access AtomSpace directly
atomspace_handle = reservoir.get_atomspace()
reservoir_atoms = reservoir.get_symbolic_representation()
```

### Bidirectional Integration

```python
# Start with Python reservoir
from reservoirpy.nodes import Reservoir, Ridge

py_reservoir = Reservoir(100, lr=0.3, sr=0.9)
py_readout = Ridge(ridge=1e-6)
py_model = py_reservoir >> py_readout

# Train in Python
py_model.fit(X_train, y_train)

# Convert to C++ AtomSpace representation
cpp_reservoir = py_reservoir.to_atomspace_reservoir()

# Continue processing in C++
cpp_reservoir.analyze_dynamics()
symbolic_patterns = cpp_reservoir.extract_patterns()
```

## Best Practices

### Performance Optimization

1. **Use C++ for computation-intensive tasks**
2. **Store only essential information in AtomSpace**
3. **Batch operations when possible**
4. **Use appropriate precision (float vs double)**

### Memory Management

```cpp
// Proper resource management
{
    auto atomspace = createAtomSpace();
    auto esn = std::make_shared<EchoStateNetwork>(atomspace, 100, 3, 1);
    
    // Use RAII for automatic cleanup
    // AtomSpace and ESN will be cleaned up automatically
}
```

### Debugging and Monitoring

```cpp
// Enable AtomSpace logging
atomspace->set_logger_level(Logger::DEBUG);

// Monitor reservoir dynamics
esn->set_debug_mode(true);

// Visualize AtomSpace content
atomspace->print(); // For debugging
```

## Examples and Tutorials

See the `examples/atomspace/` directory for complete examples:

- `basic_esn_atomspace.cpp` - Simple ESN with AtomSpace
- `symbolic_pattern_recognition.cpp` - Pattern recognition with symbolic reasoning
- `temporal_logic_sequences.cpp` - Temporal logic and sequence modeling
- `evolutionary_optimization.cpp` - Evolutionary reservoir optimization
- `python_cpp_bridge.py` - Bidirectional Python-C++ integration