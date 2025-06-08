/**
 * EmotionNode.h
 * 
 * Differential Emotion Theory Framework - C++ AtomSpace Integration
 * 
 * This header implements emotion processing nodes for symbolic AI integration,
 * providing high-performance emotion computation with AtomSpace knowledge representation.
 */

#ifndef OPENCOG_EMOTION_NODE_H
#define OPENCOG_EMOTION_NODE_H

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include <opencog/atoms/base/Node.h>
#include <opencog/atomspace/AtomSpace.h>
#include <opencog/atoms/value/FloatValue.h>
#include <opencog/atoms/atom_types/atom_types.h>

namespace opencog {
namespace reservoir {

/**
 * Basic emotions from Differential Emotion Theory
 */
enum class BasicEmotion {
    INTEREST = 0,
    JOY = 1,
    SURPRISE = 2,
    SADNESS = 3,
    ANGER = 4,
    DISGUST = 5,
    CONTEMPT = 6,
    FEAR = 7,
    SHAME = 8,
    GUILT = 9
};

/**
 * Emotion state representation
 */
struct EmotionState {
    std::vector<double> basic_emotions;
    double valence;
    double arousal;
    double intensity;
    
    EmotionState(int num_emotions = 10) 
        : basic_emotions(num_emotions, 0.0), valence(0.0), arousal(0.0), intensity(0.0) {}
};

/**
 * Differential Emotion Processor Node
 * 
 * AtomSpace-integrated emotion processing following Differential Emotion Theory.
 * Provides symbolic representation of emotional states and reasoning capabilities.
 */
class EmotionNode : public Node {
public:
    EmotionNode(AtomSpacePtr atomspace, const std::string& name = "EmotionNode");
    virtual ~EmotionNode() = default;
    
    // Core emotion processing
    EmotionState processInput(const std::vector<double>& input);
    void updateEmotionState(const EmotionState& new_state);
    EmotionState getCurrentState() const { return current_state_; }
    
    // AtomSpace integration
    void storeEmotionInAtomSpace(const EmotionState& state);
    EmotionState loadEmotionFromAtomSpace();
    Handle createEmotionAtom(const EmotionState& state);
    
    // Symbolic reasoning
    Handle queryDominantEmotion();
    Handle queryEmotionByType(BasicEmotion emotion_type);
    std::vector<Handle> queryEmotionHistory(int time_steps = 10);
    
    // Configuration
    void setEmotionDimensions(int dimensions);
    void setTemporalDynamics(bool enabled);
    void setValenceArousalEnabled(bool enabled);
    void setDecayRate(double rate);
    void setAdaptationRate(double rate);
    
    // Analysis
    BasicEmotion getDominantEmotion() const;
    double getEmotionIntensity(BasicEmotion emotion) const;
    std::pair<double, double> getValenceArousal() const;
    
    // Utility
    static std::string emotionToString(BasicEmotion emotion);
    static BasicEmotion stringToEmotion(const std::string& emotion_str);

private:
    AtomSpacePtr atomspace_;
    EmotionState current_state_;
    EmotionState previous_state_;
    
    // Configuration parameters
    int emotion_dimensions_;
    bool temporal_dynamics_enabled_;
    bool valence_arousal_enabled_;
    double decay_rate_;
    double adaptation_rate_;
    
    // Processing weights (initialized during first use)
    std::vector<std::vector<double>> emotion_weights_;
    std::vector<double> valence_weights_;
    std::vector<double> arousal_weights_;
    bool weights_initialized_;
    
    // AtomSpace keys for emotion storage
    Handle emotion_state_key_;
    Handle emotion_history_key_;
    Handle emotion_valence_key_;
    Handle emotion_arousal_key_;
    
    // Internal processing methods
    void initializeWeights(int input_size);
    std::vector<double> applyEmotionDynamics(const std::vector<double>& raw_emotions);
    EmotionState applyTemporalDynamics(const EmotionState& current_emotion);
    double softmax(const std::vector<double>& values, int index);
    void normalizeEmotions(std::vector<double>& emotions);
    
    // AtomSpace utility methods
    void initializeAtomSpaceKeys();
    Handle getOrCreateEmotionNode(const std::string& emotion_name);
};

} // namespace reservoir
} // namespace opencog

#endif // OPENCOG_EMOTION_NODE_H