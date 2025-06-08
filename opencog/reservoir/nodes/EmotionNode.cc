/**
 * EmotionNode.cc
 * 
 * Implementation of Differential Emotion Theory Framework for OpenCog AtomSpace
 */

#include "EmotionNode.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/value/LinkValue.h>

namespace opencog {
namespace reservoir {

EmotionNode::EmotionNode(AtomSpacePtr atomspace, const std::string& name)
    : Node(CONCEPT_NODE, name), atomspace_(atomspace), 
      emotion_dimensions_(10), temporal_dynamics_enabled_(true),
      valence_arousal_enabled_(true), decay_rate_(0.1), adaptation_rate_(0.05),
      weights_initialized_(false) {
    
    current_state_ = EmotionState(emotion_dimensions_);
    previous_state_ = EmotionState(emotion_dimensions_);
    
    initializeAtomSpaceKeys();
}

EmotionState EmotionNode::processInput(const std::vector<double>& input) {
    if (!weights_initialized_) {
        initializeWeights(input.size());
    }
    
    // Compute basic emotion activations
    std::vector<double> emotion_raw(emotion_dimensions_, 0.0);
    for (int i = 0; i < emotion_dimensions_; ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            emotion_raw[i] += input[j] * emotion_weights_[i][j];
        }
    }
    
    // Apply emotion dynamics
    auto emotions = applyEmotionDynamics(emotion_raw);
    
    EmotionState new_state(emotion_dimensions_);
    new_state.basic_emotions = emotions;
    
    // Compute valence and arousal if enabled
    if (valence_arousal_enabled_) {
        new_state.valence = 0.0;
        new_state.arousal = 0.0;
        
        for (size_t i = 0; i < input.size(); ++i) {
            new_state.valence += input[i] * valence_weights_[i];
            new_state.arousal += input[i] * arousal_weights_[i];
        }
        
        // Apply sigmoid to keep valence/arousal in [-1, 1] range
        new_state.valence = 2.0 / (1.0 + std::exp(-new_state.valence)) - 1.0;
        new_state.arousal = 2.0 / (1.0 + std::exp(-new_state.arousal)) - 1.0;
    }
    
    // Calculate overall intensity
    new_state.intensity = 0.0;
    for (double emotion : new_state.basic_emotions) {
        new_state.intensity += emotion * emotion;
    }
    new_state.intensity = std::sqrt(new_state.intensity);
    
    // Apply temporal dynamics if enabled
    if (temporal_dynamics_enabled_) {
        new_state = applyTemporalDynamics(new_state);
    }
    
    updateEmotionState(new_state);
    return new_state;
}

void EmotionNode::updateEmotionState(const EmotionState& new_state) {
    previous_state_ = current_state_;
    current_state_ = new_state;
    
    // Store in AtomSpace
    storeEmotionInAtomSpace(new_state);
}

void EmotionNode::storeEmotionInAtomSpace(const EmotionState& state) {
    // Create emotion values
    auto emotion_values = std::make_shared<FloatValue>(state.basic_emotions);
    setValue(emotion_state_key_, emotion_values);
    
    if (valence_arousal_enabled_) {
        auto valence_value = std::make_shared<FloatValue>(std::vector<double>{state.valence});
        auto arousal_value = std::make_shared<FloatValue>(std::vector<double>{state.arousal});
        setValue(emotion_valence_key_, valence_value);
        setValue(emotion_arousal_key_, arousal_value);
    }
    
    // Store emotion concepts symbolically
    BasicEmotion dominant = getDominantEmotion();
    std::string emotion_name = emotionToString(dominant);
    Handle emotion_concept = getOrCreateEmotionNode(emotion_name);
    
    // Create temporal link for emotion history
    Handle evaluation_link = atomspace_->add_link(EVALUATION_LINK,
        emotion_concept,
        atomspace_->add_node(NUMBER_NODE, std::to_string(state.intensity))
    );
}

BasicEmotion EmotionNode::getDominantEmotion() const {
    auto max_it = std::max_element(current_state_.basic_emotions.begin(),
                                   current_state_.basic_emotions.end());
    int index = std::distance(current_state_.basic_emotions.begin(), max_it);
    return static_cast<BasicEmotion>(index);
}

double EmotionNode::getEmotionIntensity(BasicEmotion emotion) const {
    int index = static_cast<int>(emotion);
    if (index >= 0 && index < static_cast<int>(current_state_.basic_emotions.size())) {
        return current_state_.basic_emotions[index];
    }
    return 0.0;
}

std::pair<double, double> EmotionNode::getValenceArousal() const {
    return {current_state_.valence, current_state_.arousal};
}

void EmotionNode::initializeWeights(int input_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);
    
    // Initialize emotion weights
    emotion_weights_.resize(emotion_dimensions_);
    for (int i = 0; i < emotion_dimensions_; ++i) {
        emotion_weights_[i].resize(input_size);
        for (int j = 0; j < input_size; ++j) {
            emotion_weights_[i][j] = dist(gen);
        }
    }
    
    // Initialize valence/arousal weights if enabled
    if (valence_arousal_enabled_) {
        valence_weights_.resize(input_size);
        arousal_weights_.resize(input_size);
        for (int i = 0; i < input_size; ++i) {
            valence_weights_[i] = dist(gen);
            arousal_weights_[i] = dist(gen);
        }
    }
    
    weights_initialized_ = true;
}

std::vector<double> EmotionNode::applyEmotionDynamics(const std::vector<double>& raw_emotions) {
    std::vector<double> emotions = raw_emotions;
    
    // Apply softmax for competition between emotions
    double max_val = *std::max_element(emotions.begin(), emotions.end());
    double sum = 0.0;
    
    for (double& emotion : emotions) {
        emotion = std::exp(emotion - max_val);
        sum += emotion;
    }
    
    for (double& emotion : emotions) {
        emotion /= sum;
    }
    
    // Apply emotion-specific constraints based on differential emotion theory
    // Interest and joy are approach emotions (positive)
    if (emotions.size() > 0) emotions[0] = std::max(0.0, emotions[0]); // interest
    if (emotions.size() > 1) emotions[1] = std::max(0.0, emotions[1]); // joy
    
    // Fear, sadness, anger are avoidance emotions  
    if (emotions.size() > 7) emotions[7] = std::max(0.0, emotions[7]); // fear
    if (emotions.size() > 3) emotions[3] = std::max(0.0, emotions[3]); // sadness
    if (emotions.size() > 4) emotions[4] = std::max(0.0, emotions[4]); // anger
    
    return emotions;
}

EmotionState EmotionNode::applyTemporalDynamics(const EmotionState& current_emotion) {
    EmotionState result = current_emotion;
    
    // Apply decay to previous state
    for (size_t i = 0; i < result.basic_emotions.size(); ++i) {
        double decayed_previous = previous_state_.basic_emotions[i] * (1.0 - decay_rate_);
        double adapted_current = current_emotion.basic_emotions[i] * adaptation_rate_;
        result.basic_emotions[i] = decayed_previous + adapted_current;
    }
    
    // Normalize emotions to maintain sum constraint
    normalizeEmotions(result.basic_emotions);
    
    // Apply temporal dynamics to valence/arousal
    if (valence_arousal_enabled_) {
        result.valence = previous_state_.valence * (1.0 - decay_rate_) + 
                        current_emotion.valence * adaptation_rate_;
        result.arousal = previous_state_.arousal * (1.0 - decay_rate_) +
                        current_emotion.arousal * adaptation_rate_;
    }
    
    return result;
}

void EmotionNode::normalizeEmotions(std::vector<double>& emotions) {
    double sum = 0.0;
    for (double emotion : emotions) {
        sum += emotion;
    }
    
    if (sum > 0.0) {
        for (double& emotion : emotions) {
            emotion /= sum;
        }
    }
}

void EmotionNode::initializeAtomSpaceKeys() {
    emotion_state_key_ = atomspace_->add_node(PREDICATE_NODE, "emotion_state");
    emotion_history_key_ = atomspace_->add_node(PREDICATE_NODE, "emotion_history");
    emotion_valence_key_ = atomspace_->add_node(PREDICATE_NODE, "emotion_valence");
    emotion_arousal_key_ = atomspace_->add_node(PREDICATE_NODE, "emotion_arousal");
}

Handle EmotionNode::getOrCreateEmotionNode(const std::string& emotion_name) {
    return atomspace_->add_node(CONCEPT_NODE, emotion_name);
}

std::string EmotionNode::emotionToString(BasicEmotion emotion) {
    switch (emotion) {
        case BasicEmotion::INTEREST: return "interest";
        case BasicEmotion::JOY: return "joy";
        case BasicEmotion::SURPRISE: return "surprise";
        case BasicEmotion::SADNESS: return "sadness";
        case BasicEmotion::ANGER: return "anger";
        case BasicEmotion::DISGUST: return "disgust";
        case BasicEmotion::CONTEMPT: return "contempt";
        case BasicEmotion::FEAR: return "fear";
        case BasicEmotion::SHAME: return "shame";
        case BasicEmotion::GUILT: return "guilt";
        default: return "unknown";
    }
}

BasicEmotion EmotionNode::stringToEmotion(const std::string& emotion_str) {
    if (emotion_str == "interest") return BasicEmotion::INTEREST;
    if (emotion_str == "joy") return BasicEmotion::JOY;
    if (emotion_str == "surprise") return BasicEmotion::SURPRISE;
    if (emotion_str == "sadness") return BasicEmotion::SADNESS;
    if (emotion_str == "anger") return BasicEmotion::ANGER;
    if (emotion_str == "disgust") return BasicEmotion::DISGUST;
    if (emotion_str == "contempt") return BasicEmotion::CONTEMPT;
    if (emotion_str == "fear") return BasicEmotion::FEAR;
    if (emotion_str == "shame") return BasicEmotion::SHAME;
    if (emotion_str == "guilt") return BasicEmotion::GUILT;
    return BasicEmotion::INTEREST; // default
}

} // namespace reservoir
} // namespace opencog