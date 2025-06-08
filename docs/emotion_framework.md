# Differential Emotion Theory Framework

## Overview

The Differential Emotion Theory Framework provides affective computing capabilities for ReservoirCogs, implementing Differential Emotion Theory for emotionally aware AI systems. This framework integrates with OpenCog AtomSpace symbolic reasoning and reservoir computing core algorithms.

## Key Features

- **Basic Emotion Processing**: Implements the 10 basic emotions from Differential Emotion Theory:
  - Interest, Joy, Surprise, Sadness, Anger, Disgust, Contempt, Fear, Shame, Guilt
- **Valence-Arousal Modeling**: Dimensional emotion representation
- **Temporal Dynamics**: Emotion state evolution over time with decay and adaptation
- **Reservoir Integration**: Emotion-aware reservoir computing with bidirectional coupling
- **AtomSpace Integration**: Symbolic emotion representation and reasoning (C++)
- **Hybrid Architecture**: Neural-symbolic fusion for explainable affective computing

## Components

### Python API

#### DifferentialEmotionProcessor
Core emotion processing node implementing Differential Emotion Theory.

```python
from reservoirpy.nodes import DifferentialEmotionProcessor

processor = DifferentialEmotionProcessor(
    emotion_dimensions=10,
    valence_arousal=True,
    temporal_dynamics=True
)

# Process emotional stimulus
emotion_state = processor.forward(stimulus)
dominant_emotion, intensity = processor.get_dominant_emotion()
valence, arousal = processor.get_valence_arousal()
```

#### EmotionReservoir
Emotion-aware reservoir computing node that integrates emotional processing with reservoir dynamics.

```python
from reservoirpy.nodes import EmotionReservoir

emotion_reservoir = EmotionReservoir(
    units=100,
    emotion_integration=0.2,
    emotion_dimensions=10,
    emotion_feedback=True
)

# Process with emotion-reservoir coupling
combined_state = emotion_reservoir.forward(input_data)
reservoir_state = emotion_reservoir.get_reservoir_state()
emotion_state = emotion_reservoir.get_emotion_state()
```

### C++ AtomSpace Integration

#### EmotionNode
High-performance emotion processing with symbolic AtomSpace representation.

```cpp
#include <opencog/reservoir/nodes/EmotionNode.h>

auto atomspace = std::make_shared<AtomSpace>();
auto emotion_node = std::make_shared<EmotionNode>(atomspace);

// Process emotional input
EmotionState state = emotion_node->processInput(input_vector);

// Query symbolic emotion representations
Handle dominant_emotion = emotion_node->queryDominantEmotion();
std::vector<Handle> emotion_history = emotion_node->queryEmotionHistory();
```

## Integration Points

The framework integrates with existing ReservoirCogs components:

- **✅ OpenCog AtomSpace**: Symbolic emotion representation and reasoning
- **✅ Reservoir Computing**: Emotion-modulated reservoir dynamics  
- **✅ GraphRAG**: Emotion-aware knowledge processing
- **✅ Codestral AI**: Affective computing explanations
- **✅ C++ Backend**: High-performance emotion computation
- **✅ Python API**: Flexible emotion processing interface
- **✅ ReservoirChat**: Emotionally aware conversational AI

## Theoretical Foundation

### Differential Emotion Theory

Differential Emotion Theory (DET) by Carroll Izard identifies discrete basic emotions that serve specific adaptive functions:

1. **Interest**: Motivates exploration and learning
2. **Joy**: Reinforces positive experiences and social bonding
3. **Surprise**: Alerts to novel stimuli and interrupts ongoing processes
4. **Sadness**: Motivates help-seeking and social support
5. **Anger**: Mobilizes energy for overcoming obstacles
6. **Disgust**: Avoids contamination and harmful substances
7. **Contempt**: Establishes social hierarchies and boundaries
8. **Fear**: Motivates escape from danger
9. **Shame**: Regulates social behavior and promotes conformity
10. **Guilt**: Motivates corrective action and moral behavior

### Dimensional Models

The framework also incorporates dimensional emotion models:

- **Valence**: Positive (pleasant) vs. negative (unpleasant) emotional content
- **Arousal**: High vs. low activation/energy level

### Temporal Dynamics

Emotions evolve over time according to:
- **Decay**: Gradual reduction in emotion intensity
- **Adaptation**: Integration of new emotional stimuli
- **Momentum**: Influence of previous emotional states

## Usage Examples

### Basic Emotion Processing

```python
import numpy as np
from reservoirpy.nodes import DifferentialEmotionProcessor

# Create emotion processor
processor = DifferentialEmotionProcessor()

# Different emotional stimuli
stimuli = {
    'joy': np.array([0.8, 0.2, 0.1]),      # High positive
    'fear': np.array([-0.8, 0.8, 0.3]),    # High negative arousal
    'sadness': np.array([-0.3, -0.2, 0.1]), # Low negative
    'interest': np.array([0.3, 0.0, 0.5])   # Moderate positive + novelty
}

for emotion_name, stimulus in stimuli.items():
    processor.forward(stimulus)
    dominant, intensity = processor.get_dominant_emotion()
    print(f"{emotion_name}: {dominant} ({intensity:.3f})")
```

### Emotion-Aware Reservoir Computing

```python
from reservoirpy.nodes import EmotionReservoir, Ridge
from reservoirpy import Model

# Create emotion-aware model
reservoir = EmotionReservoir(units=100, emotion_integration=0.2)
readout = Ridge()
model = Model(reservoir, readout)

# Train on emotional time series
model.fit(emotional_sequences, targets)

# Analyze emotional states during prediction
predictions = model.run(test_data)
emotion_timeline = [reservoir.get_dominant_emotion() 
                   for _ in test_data]
```

### C++ AtomSpace Integration

```cpp
// Create emotion-aware AtomSpace system
auto atomspace = std::make_shared<AtomSpace>();
auto emotion_node = std::make_shared<EmotionNode>(atomspace);

// Process emotional inputs
std::vector<double> input = {0.5, 0.3, 0.2};
EmotionState state = emotion_node->processInput(input);

// Query stored emotional knowledge
Handle query = atomspace->add_link(BIND_LINK,
    // Query for patterns in emotion history
);
HandleSeq results = query->execute(atomspace);
```

## Research Applications

- **Affective Computing**: Building emotionally intelligent AI systems
- **Human-Computer Interaction**: Natural emotion recognition and response
- **Cognitive Modeling**: Simulating human emotional processes
- **Social Robotics**: Emotionally appropriate robot behavior
- **Mental Health**: Computational models of emotional disorders
- **Education**: Adaptive learning systems with emotional awareness

## Performance Characteristics

- **Python**: Flexible prototyping and experimentation
- **C++**: High-performance production systems
- **AtomSpace**: Symbolic reasoning and knowledge representation
- **Real-time**: Suitable for online emotional processing
- **Scalable**: Efficient computation for large-scale systems

## Future Extensions

- **Multi-modal Emotion Recognition**: Integration with sensory inputs
- **Cultural Emotion Models**: Cross-cultural emotion variations
- **Emotion Regulation**: Active emotion management strategies
- **Social Emotion Dynamics**: Group emotion interactions
- **Neurobiological Models**: Brain-inspired emotion processing

## References

1. Izard, C. E. (2007). Basic emotions, natural kinds, emotion schemas, and a new paradigm.
2. Russell, J. A. (1980). A circumplex model of affect.
3. Ekman, P. (1992). An argument for basic emotions.
4. Barrett, L. F. (2006). Are emotions natural kinds?

## Getting Started

See the examples in `examples/differential_emotion_example.py` and 
`examples/atomspace/emotion_atomspace_example.cpp` for complete demonstrations
of the framework capabilities.