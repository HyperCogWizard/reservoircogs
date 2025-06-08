# ReservoirCogs Examples

This directory contains examples demonstrating the integration of reservoir computing with OpenCog AtomSpace.

## C++ Examples

### basic_esn_atomspace.cpp
Basic Echo State Network with AtomSpace integration. Shows:
- Creating ESN with AtomSpace backend
- Training with reservoir algorithms
- Storing knowledge in symbolic form
- Performance analysis and metrics

To build and run:
```bash
cd build
make basic_esn_example
./basic_esn_example
```

### emotion_atomspace_example.cpp
Differential Emotion Theory Framework with AtomSpace integration. Shows:
- Creating emotion-aware processors with symbolic representation
- Processing emotional stimuli with temporal dynamics
- Storing emotion states and history in AtomSpace
- Querying emotional knowledge symbolically

To build and run:
```bash
cd build
make emotion_example
./emotion_example
```

### symbolic_pattern_recognition.cpp
Advanced pattern recognition using symbolic reasoning:
- Temporal sequence analysis
- Pattern extraction as AtomSpace concepts
- Symbolic query and reasoning
- Knowledge graph construction

### temporal_logic_sequences.cpp
Demonstrates temporal logic integration:
- Time-indexed reservoir states
- Temporal relationship modeling
- Sequential pattern reasoning
- Event detection and prediction

### evolutionary_optimization.cpp
Evolutionary optimization of reservoir topologies:
- Population-based optimization
- Fitness evaluation with symbolic metrics
- Mutation and crossover operators
- Best topology selection and analysis

## Python Examples

### python_cpp_bridge.py
Bidirectional Python-C++ integration:
- Converting Python reservoirs to C++ AtomSpace
- Using C++ performance with Python flexibility
- Hybrid neural-symbolic reasoning
- Symbolic knowledge extraction

To run:
```bash
python examples/atomspace/python_cpp_bridge.py
```

## Building Examples

### Prerequisites
- OpenCog AtomSpace development libraries
- CMake 3.16+
- C++17 compatible compiler
- Python 3.8+ (for Python examples)

### Build Instructions
```bash
# From repository root
mkdir build && cd build
cmake -DUNIT_TESTS=ON ..
make -j4

# Run C++ examples
./examples/basic_esn_example
./examples/symbolic_pattern_example
./examples/temporal_logic_example
./examples/evolutionary_optimization_example

# Run Python examples
cd ..
python examples/atomspace/python_cpp_bridge.py
```

## Key Concepts Demonstrated

### AtomSpace Integration
- Custom atom types for reservoir computing
- Symbolic representation of neural states
- Knowledge storage and retrieval
- Graph-based reasoning

### Hybrid AI Architecture
- Neural computation with symbolic reasoning
- Explainable AI through symbolic extraction
- Temporal logic and sequence modeling
- Concept formation from patterns

### Performance Optimization
- High-performance C++ computation
- Efficient AtomSpace operations
- Memory management and resource cleanup
- Scalable reservoir architectures

### Research Applications
- Time series prediction with explanation
- Pattern recognition and classification
- Anomaly detection with symbolic rules
- Cognitive modeling and reasoning

## Further Reading
- [AtomSpace Integration Guide](../docs/user_guide/atomspace_integration.md)
- [User Guide](../docs/user_guide/index.md)
- [Getting Started](../docs/getting_started.md)