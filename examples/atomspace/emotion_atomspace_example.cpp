/**
 * emotion_atomspace_example.cpp
 * 
 * Example demonstrating Differential Emotion Theory Framework with AtomSpace integration
 */

#include <iostream>
#include <vector>
#include <memory>

#include <opencog/atomspace/AtomSpace.h>
#include "EmotionNode.h"

using namespace opencog;
using namespace opencog::reservoir;

int main() {
    std::cout << "=== Differential Emotion Theory Framework - C++ Example ===\n\n";
    
    // Create AtomSpace
    auto atomspace = std::make_shared<AtomSpace>();
    
    // Create emotion processor
    auto emotion_node = std::make_shared<EmotionNode>(atomspace, "MainEmotionProcessor");
    
    std::cout << "1. Created emotion processor with AtomSpace integration\n";
    
    // Configure emotion processor
    emotion_node->setEmotionDimensions(10);
    emotion_node->setTemporalDynamics(true);
    emotion_node->setValenceArousalEnabled(true);
    emotion_node->setDecayRate(0.1);
    emotion_node->setAdaptationRate(0.05);
    
    std::cout << "2. Configured emotion processor parameters\n";
    
    // Test different emotional inputs
    std::vector<std::vector<double>> test_inputs = {
        {0.8, 0.2, 0.1},     // High positive - should trigger joy
        {-0.8, 0.8, 0.3},    // High negative arousal - should trigger fear/anger
        {-0.3, -0.2, 0.1},   // Low negative - should trigger sadness  
        {0.3, 0.0, 0.5},     // Moderate positive with novelty - should trigger interest
        {0.0, 0.0, 0.0}      // Neutral input
    };
    
    std::vector<std::string> input_descriptions = {
        "High positive stimulus (joy trigger)",
        "High negative arousal (fear/anger trigger)", 
        "Low negative stimulus (sadness trigger)",
        "Moderate positive with novelty (interest trigger)",
        "Neutral input"
    };
    
    std::cout << "3. Processing emotional inputs:\n\n";
    
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        std::cout << "Input " << (i+1) << ": " << input_descriptions[i] << "\n";
        std::cout << "  Values: [";
        for (size_t j = 0; j < test_inputs[i].size(); ++j) {
            std::cout << test_inputs[i][j];
            if (j < test_inputs[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        
        // Process input
        EmotionState result = emotion_node->processInput(test_inputs[i]);
        
        // Get dominant emotion
        BasicEmotion dominant = emotion_node->getDominantEmotion();
        double intensity = emotion_node->getEmotionIntensity(dominant);
        
        // Get valence and arousal
        auto valence_arousal = emotion_node->getValenceArousal();
        
        std::cout << "  Result:\n";
        std::cout << "    Dominant emotion: " << EmotionNode::emotionToString(dominant) 
                  << " (intensity: " << intensity << ")\n";
        std::cout << "    Valence: " << valence_arousal.first 
                  << ", Arousal: " << valence_arousal.second << "\n";
        
        // Show top 3 emotions
        std::cout << "    Emotion vector: ";
        for (size_t j = 0; j < result.basic_emotions.size() && j < 3; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << EmotionNode::emotionToString(static_cast<BasicEmotion>(j)) 
                      << "(" << result.basic_emotions[j] << ")";
        }
        std::cout << "\n\n";
    }
    
    std::cout << "4. Demonstrating AtomSpace integration:\n";
    
    // Query AtomSpace for stored emotion information
    std::cout << "  AtomSpace contains " << atomspace->get_size() << " atoms\n";
    
    // The emotion states have been automatically stored in AtomSpace
    // during processing via the storeEmotionInAtomSpace method
    
    std::cout << "  Emotion concepts have been created and stored\n";
    std::cout << "  Temporal emotion history is maintained\n";
    std::cout << "  Symbolic reasoning about emotions is now possible\n";
    
    std::cout << "\n5. Temporal dynamics demonstration:\n";
    
    // Process a sequence to show temporal dynamics
    std::vector<double> sequence_input = {0.5, 0.2, 0.1};
    
    for (int step = 0; step < 5; ++step) {
        // Slightly modify input each step
        sequence_input[0] += (step - 2) * 0.1;
        
        EmotionState state = emotion_node->processInput(sequence_input);
        BasicEmotion dominant = emotion_node->getDominantEmotion();
        double intensity = emotion_node->getEmotionIntensity(dominant);
        
        std::cout << "  Step " << step+1 << ": " 
                  << EmotionNode::emotionToString(dominant)
                  << " (" << intensity << ")\n";
    }
    
    std::cout << "\n=== Example completed successfully! ===\n";
    std::cout << "\nThis example demonstrated:\n";
    std::cout << "- Differential emotion theory processing\n";
    std::cout << "- AtomSpace integration for symbolic emotion representation\n";
    std::cout << "- Temporal emotion dynamics with decay and adaptation\n";
    std::cout << "- Valence-arousal dimensional modeling\n";
    std::cout << "- High-performance C++ emotion computation\n";
    
    return 0;
}