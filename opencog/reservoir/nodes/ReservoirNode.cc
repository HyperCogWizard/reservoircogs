/*
 * ReservoirNode.cc
 * 
 * Implementation of reservoir computing nodes with AtomSpace integration
 */

#include "ReservoirNode.h"
#include "../types/reservoir_types.h"
#include <opencog/atoms/base/Node.h>
#include <opencog/atoms/value/FloatValue.h>
#include <random>
#include <algorithm>
#include <cmath>

using namespace opencog;
using namespace opencog::reservoir;

ReservoirNode::ReservoirNode(AtomSpacePtr as, size_t reservoir_size)
    : _atomspace(as), _size(reservoir_size)
{
    // Create AtomSpace handle for this reservoir
    _self_handle = _atomspace->add_node(RESERVOIR_NODE, "reservoir_" + std::to_string(reservoir_size));
    
    // Initialize state and weights
    _state.resize(_size, 0.0);
    _weights.resize(_size, std::vector<double>(_size, 0.0));
    
    initialize_weights();
    normalize_weights(0.9); // Default spectral radius
}

std::vector<double> ReservoirNode::update(const std::vector<double>& input)
{
    std::vector<double> new_state(_size, 0.0);
    
    // Compute new state: new_state = tanh(W * state + W_in * input)
    for (size_t i = 0; i < _size; ++i) {
        double sum = 0.0;
        
        // Internal connections
        for (size_t j = 0; j < _size; ++j) {
            sum += _weights[i][j] * _state[j];
        }
        
        // Input connections (simplified - assumes direct mapping for now)
        if (i < input.size()) {
            sum += input[i];
        }
        
        new_state[i] = activation_function(sum);
    }
    
    _state = new_state;
    store_state_to_atomspace();
    
    return _state;
}

void ReservoirNode::reset_state()
{
    std::fill(_state.begin(), _state.end(), 0.0);
    store_state_to_atomspace();
}

void ReservoirNode::store_state_to_atomspace()
{
    // Create FloatValue to store current state
    ValuePtr state_value = createFloatValue(_state);
    _self_handle->setValue(_atomspace->add_node(PREDICATE_NODE, "state"), state_value);
}

void ReservoirNode::set_weights_from_atomspace(Handle weights_handle)
{
    // Extract weights from AtomSpace and update internal weights
    // This would typically involve parsing a specific atom structure
    // For now, implementing a placeholder
}

double ReservoirNode::activation_function(double x) const
{
    return std::tanh(x); // Default activation function
}

void ReservoirNode::initialize_weights(double density)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    std::uniform_real_distribution<> sparse_dis(0.0, 1.0);
    
    for (size_t i = 0; i < _size; ++i) {
        for (size_t j = 0; j < _size; ++j) {
            if (sparse_dis(gen) < density) {
                _weights[i][j] = dis(gen);
            } else {
                _weights[i][j] = 0.0;
            }
        }
    }
}

void ReservoirNode::normalize_weights(double spectral_radius)
{
    // Simplified spectral radius normalization
    // In a full implementation, this would compute the actual spectral radius
    double max_weight = 0.0;
    for (const auto& row : _weights) {
        for (double w : row) {
            max_weight = std::max(max_weight, std::abs(w));
        }
    }
    
    if (max_weight > 0) {
        double scale = spectral_radius / max_weight;
        for (auto& row : _weights) {
            for (double& w : row) {
                w *= scale;
            }
        }
    }
}

double ReservoirNode::get_spectral_radius() const
{
    // Placeholder - would compute actual spectral radius using eigenvalues
    return 0.9;
}

// EchoStateNetwork implementation
EchoStateNetwork::EchoStateNetwork(AtomSpacePtr as, size_t reservoir_size, 
                                   size_t input_size, size_t output_size)
    : ReservoirNode(as, reservoir_size), _leaking_rate(1.0)
{
    // Initialize input and output weight matrices
    _input_weights.resize(reservoir_size, std::vector<double>(input_size, 0.0));
    _output_weights.resize(output_size, std::vector<double>(reservoir_size, 0.0));
    _feedback_weights.resize(reservoir_size, std::vector<double>(output_size, 0.0));
    
    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    for (auto& row : _input_weights) {
        for (double& w : row) {
            w = dis(gen);
        }
    }
}

std::vector<double> EchoStateNetwork::update(const std::vector<double>& input)
{
    std::vector<double> new_state(_size, 0.0);
    
    // ESN update equation with leaking rate
    for (size_t i = 0; i < _size; ++i) {
        double sum = 0.0;
        
        // Internal connections
        for (size_t j = 0; j < _size; ++j) {
            sum += _weights[i][j] * _state[j];
        }
        
        // Input connections
        for (size_t j = 0; j < input.size() && j < _input_weights[i].size(); ++j) {
            sum += _input_weights[i][j] * input[j];
        }
        
        // Apply leaking rate integration
        new_state[i] = (1.0 - _leaking_rate) * _state[i] + 
                       _leaking_rate * activation_function(sum);
    }
    
    _state = new_state;
    store_state_to_atomspace();
    
    return _state;
}

void EchoStateNetwork::train_readout(const std::vector<std::vector<double>>& inputs,
                                    const std::vector<std::vector<double>>& targets)
{
    // Simplified ridge regression for readout training
    // In a full implementation, this would use proper linear algebra libraries
    
    // Collect states for all input sequences
    std::vector<std::vector<double>> states;
    reset_state();
    
    for (const auto& input_seq : inputs) {
        update(input_seq);
        states.push_back(get_state());
    }
    
    // Train output weights using ridge regression (placeholder implementation)
    // This would typically use libraries like Eigen for proper matrix operations
}

std::vector<double> EchoStateNetwork::predict(const std::vector<double>& input)
{
    // Update reservoir state
    auto state = update(input);
    
    // Compute output
    std::vector<double> output(_output_weights.size(), 0.0);
    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < state.size() && j < _output_weights[i].size(); ++j) {
            output[i] += _output_weights[i][j] * state[j];
        }
    }
    
    return output;
}

void EchoStateNetwork::store_weights_to_atomspace()
{
    // Store all weight matrices as AtomSpace values
    // This would create structured representations of the weight matrices
    // using the custom reservoir atom types
}

void EchoStateNetwork::load_weights_from_atomspace()
{
    // Load weight matrices from AtomSpace
    // This would parse the structured representations and update internal weights
}