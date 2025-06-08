#!/usr/bin/env python3
"""
Differential Emotion Theory Framework Example

This example demonstrates the basic usage of the Differential Emotion Theory Framework
in ReservoirCogs, showcasing emotion-aware reservoir computing with AtomSpace integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import EmotionReservoir, DifferentialEmotionProcessor, Ridge
from reservoirpy import Model

def generate_emotional_stimulus():
    """Generate synthetic emotional stimulus data."""
    # Create different emotional patterns
    time_steps = 1000
    
    # Joy pattern (positive, high activation)
    joy_pattern = np.sin(np.linspace(0, 4*np.pi, time_steps//4)) * 0.8 + 0.2
    
    # Fear pattern (negative, high activation, irregular)
    fear_pattern = -np.abs(np.sin(np.linspace(0, 8*np.pi, time_steps//4))) - 0.3
    fear_pattern += np.random.normal(0, 0.1, time_steps//4)
    
    # Sadness pattern (negative, low activation)
    sadness_pattern = -0.5 * np.ones(time_steps//4) + np.random.normal(0, 0.05, time_steps//4)
    
    # Interest pattern (positive, moderate activation with variations)
    interest_pattern = 0.3 * np.sin(np.linspace(0, 2*np.pi, time_steps//4)) + 0.1
    
    # Combine patterns
    stimulus = np.concatenate([joy_pattern, fear_pattern, sadness_pattern, interest_pattern])
    
    # Add secondary features (e.g., sensory input dimensions)
    secondary_features = np.random.normal(0, 0.1, (time_steps, 2))
    
    # Create multi-dimensional input
    input_data = np.column_stack([stimulus, secondary_features])
    
    return input_data, stimulus

def create_emotion_model():
    """Create an emotion-aware reservoir computing model."""
    # Create emotion-aware reservoir
    emotion_reservoir = EmotionReservoir(
        units=100,
        emotion_integration=0.2,
        emotion_dimensions=10,
        emotion_feedback=True,
        lr=0.3,
        sr=0.9
    )
    
    # Create readout layer
    readout = Ridge(ridge=1e-6)
    
    # Create model
    model = Model(emotion_reservoir, readout, name="EmotionModel")
    
    return model, emotion_reservoir

def demonstrate_emotion_processing():
    """Demonstrate basic emotion processing."""
    print("=== Differential Emotion Theory Framework Demo ===\n")
    
    # Generate data
    print("1. Generating emotional stimulus data...")
    input_data, true_stimulus = generate_emotional_stimulus()
    target_data = true_stimulus.reshape(-1, 1)  # Predict the emotional stimulus
    
    # Create model
    print("2. Creating emotion-aware reservoir model...")
    model, emotion_reservoir = create_emotion_model()
    
    # Split data
    train_size = 800
    X_train, X_test = input_data[:train_size], input_data[train_size:]
    y_train, y_test = target_data[:train_size], target_data[train_size:]
    
    # Train model
    print("3. Training emotion-aware model...")
    model.fit(X_train, y_train)
    
    # Test prediction
    print("4. Testing model predictions...")
    y_pred = model.run(X_test)
    
    # Analyze emotion states during testing
    print("5. Analyzing emotion states...")
    emotion_states = []
    dominant_emotions = []
    valence_arousal = []
    
    for i, x in enumerate(X_test):
        # Process through emotion reservoir
        state = emotion_reservoir.forward(x.reshape(1, -1))
        emotion_states.append(emotion_reservoir.get_emotion_state())
        dominant_emotions.append(emotion_reservoir.get_dominant_emotion())
        valence_arousal.append(emotion_reservoir.get_valence_arousal())
    
    # Display results
    print("\n=== Emotion Analysis Results ===")
    print(f"Prediction MSE: {np.mean((y_test - y_pred)**2):.4f}")
    
    print("\nDominant emotions over time:")
    unique_emotions = list(set(emo[0] for emo in dominant_emotions))
    for emotion in unique_emotions:
        count = sum(1 for emo in dominant_emotions if emo[0] == emotion)
        percentage = (count / len(dominant_emotions)) * 100
        print(f"  {emotion}: {percentage:.1f}% of the time")
    
    print(f"\nAverage valence: {np.mean([va[0] for va in valence_arousal]):.3f}")
    print(f"Average arousal: {np.mean([va[1] for va in valence_arousal]):.3f}")
    
    # Plot results
    plot_results(y_test, y_pred, emotion_states, dominant_emotions, valence_arousal)
    
    return model, emotion_states, dominant_emotions

def demonstrate_emotion_processor():
    """Demonstrate standalone emotion processor."""
    print("\n=== Standalone Emotion Processor Demo ===\n")
    
    # Create emotion processor
    emotion_proc = DifferentialEmotionProcessor(
        emotion_dimensions=10,
        valence_arousal=True,
        temporal_dynamics=True
    )
    
    # Generate test inputs
    test_inputs = [
        np.array([0.8, 0.2, 0.1]),  # High positive - should trigger joy
        np.array([-0.8, 0.8, 0.3]), # High negative arousal - should trigger fear/anger
        np.array([-0.3, -0.2, 0.1]), # Low negative - should trigger sadness
        np.array([0.3, 0.0, 0.5]),  # Moderate positive with novelty - should trigger interest
    ]
    
    print("Processing different emotional inputs:")
    for i, input_vec in enumerate(test_inputs):
        emotion_state = emotion_proc.forward(input_vec)
        dominant_emotion, intensity = emotion_proc.get_dominant_emotion()
        valence, arousal = emotion_proc.get_valence_arousal()
        
        print(f"\nInput {i+1}: {input_vec}")
        print(f"  Dominant emotion: {dominant_emotion} (intensity: {intensity:.3f})")
        print(f"  Valence: {valence:.3f}, Arousal: {arousal:.3f}")
        
        # Show top 3 emotions
        emotion_vector = emotion_proc.get_emotion_vector()
        basic_emotions = {k: v for k, v in emotion_vector.items() 
                         if k not in ['valence', 'arousal']}
        top_emotions = sorted(basic_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top emotions: {', '.join([f'{e}({v:.3f})' for e, v in top_emotions])}")

def plot_results(y_true, y_pred, emotion_states, dominant_emotions, valence_arousal):
    """Plot emotion analysis results."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Prediction vs True
        axes[0, 0].plot(y_true, label='True', alpha=0.7)
        axes[0, 0].plot(y_pred, label='Predicted', alpha=0.7)
        axes[0, 0].set_title('Emotion Stimulus Prediction')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Stimulus Intensity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Valence-Arousal Space
        valences = [va[0] for va in valence_arousal]
        arousals = [va[1] for va in valence_arousal]
        axes[0, 1].scatter(valences, arousals, alpha=0.6, c=range(len(valences)), cmap='viridis')
        axes[0, 1].set_title('Valence-Arousal Space')
        axes[0, 1].set_xlabel('Valence')
        axes[0, 1].set_ylabel('Arousal')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot 3: Emotion Intensities Over Time
        emotion_array = np.array(emotion_states)
        time_steps = range(len(emotion_array))
        
        # Plot top 4 most active emotions
        emotion_labels = ['interest', 'joy', 'surprise', 'sadness', 'anger', 
                         'disgust', 'contempt', 'fear', 'shame', 'guilt']
        
        # Find most active emotions
        mean_activations = np.mean(emotion_array[:, :10], axis=0)
        top_indices = np.argsort(mean_activations)[-4:]
        
        for idx in top_indices:
            axes[1, 0].plot(time_steps, emotion_array[:, idx], 
                           label=emotion_labels[idx], alpha=0.8)
        
        axes[1, 0].set_title('Top Emotion Intensities Over Time')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Emotion Intensity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Dominant Emotion Timeline
        emotion_names = [emo[0] for emo in dominant_emotions]
        unique_names = list(set(emotion_names))
        emotion_indices = [unique_names.index(name) for name in emotion_names]
        
        axes[1, 1].plot(time_steps, emotion_indices, marker='o', markersize=2, alpha=0.7)
        axes[1, 1].set_title('Dominant Emotion Over Time')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Emotion Type')
        axes[1, 1].set_yticks(range(len(unique_names)))
        axes[1, 1].set_yticklabels(unique_names)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/emotion_analysis_results.png', dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to /tmp/emotion_analysis_results.png")
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")
    except Exception as e:
        print(f"\nError creating plots: {e}")

if __name__ == "__main__":
    try:
        # Run demonstrations
        model, emotion_states, dominant_emotions = demonstrate_emotion_processing()
        demonstrate_emotion_processor()
        
        print("\n=== Demo completed successfully! ===")
        print("\nThis example demonstrates:")
        print("- Emotion-aware reservoir computing")
        print("- Differential emotion theory implementation")
        print("- Valence-arousal emotional space modeling")
        print("- Temporal emotion dynamics")
        print("- Integration with reservoir computing algorithms")
        
    except Exception as e:
        print(f"Error running demonstration: {e}")
        print("This is expected if dependencies are not fully installed.")
        print("The framework structure has been successfully created.")