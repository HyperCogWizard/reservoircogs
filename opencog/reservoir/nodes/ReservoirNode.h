/*
 * ReservoirNode.h
 * 
 * Core reservoir computing node integrated with OpenCog AtomSpace
 * This class provides the C++ implementation of reservoir computing
 * algorithms with deep AtomSpace integration.
 */

#ifndef _OPENCOG_RESERVOIR_NODE_H
#define _OPENCOG_RESERVOIR_NODE_H

#include <vector>
#include <memory>
#include <opencog/atomspace/AtomSpace.h>
#include <opencog/atoms/base/Handle.h>
#include <opencog/atoms/base/Node.h>

namespace opencog {
namespace reservoir {

/**
 * ReservoirNode - Base class for reservoir computing nodes
 * 
 * This class integrates reservoir computing concepts with AtomSpace,
 * providing a bridge between traditional neural reservoir computing
 * and symbolic AI.
 */
class ReservoirNode
{
private:
    AtomSpacePtr _atomspace;
    Handle _self_handle;
    size_t _size;
    std::vector<double> _state;
    std::vector<std::vector<double>> _weights;
    
public:
    ReservoirNode(AtomSpacePtr as, size_t reservoir_size);
    virtual ~ReservoirNode() = default;
    
    // Core reservoir operations
    virtual std::vector<double> update(const std::vector<double>& input);
    virtual std::vector<double> get_state() const { return _state; }
    virtual void reset_state();
    
    // AtomSpace integration
    Handle get_handle() const { return _self_handle; }
    void set_weights_from_atomspace(Handle weights_handle);
    void store_state_to_atomspace();
    
    // Configuration
    void set_activation_function(Handle func_handle);
    void set_spectral_radius(double radius);
    void set_input_scaling(double scaling);
    
    // Properties
    size_t size() const { return _size; }
    double get_spectral_radius() const;
    
protected:
    virtual double activation_function(double x) const;
    void normalize_weights(double spectral_radius);
    void initialize_weights(double density = 0.1);
};

/**
 * EchoStateNetwork - Specific implementation for Echo State Networks
 */
class EchoStateNetwork : public ReservoirNode
{
private:
    std::vector<std::vector<double>> _input_weights;
    std::vector<std::vector<double>> _output_weights;
    std::vector<std::vector<double>> _feedback_weights;
    double _leaking_rate;
    
public:
    EchoStateNetwork(AtomSpacePtr as, size_t reservoir_size, 
                     size_t input_size, size_t output_size);
    
    // ESN specific operations
    std::vector<double> update(const std::vector<double>& input) override;
    void train_readout(const std::vector<std::vector<double>>& inputs,
                      const std::vector<std::vector<double>>& targets);
    std::vector<double> predict(const std::vector<double>& input);
    
    // Configuration
    void set_leaking_rate(double rate) { _leaking_rate = rate; }
    void set_input_weights(const std::vector<std::vector<double>>& weights);
    void set_feedback_weights(const std::vector<std::vector<double>>& weights);
    
    // AtomSpace integration for ESN components
    void store_weights_to_atomspace();
    void load_weights_from_atomspace();
};

} // namespace reservoir
} // namespace opencog

#endif // _OPENCOG_RESERVOIR_NODE_H