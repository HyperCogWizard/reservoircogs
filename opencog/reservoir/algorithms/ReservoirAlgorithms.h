/*
 * ReservoirAlgorithms.h
 * 
 * Core reservoir computing algorithms integrated with AtomSpace
 * This provides high-level algorithms for training and optimization
 */

#ifndef _OPENCOG_RESERVOIR_ALGORITHMS_H
#define _OPENCOG_RESERVOIR_ALGORITHMS_H

#include <vector>
#include <memory>
#include <opencog/atomspace/AtomSpace.h>
#include "../nodes/ReservoirNode.h"

namespace opencog {
namespace reservoir {
namespace algorithms {

/**
 * ReservoirTrainer - Training algorithms for reservoir computing
 */
class ReservoirTrainer
{
private:
    AtomSpacePtr _atomspace;
    
public:
    ReservoirTrainer(AtomSpacePtr as) : _atomspace(as) {}
    
    // Training methods
    void train_esn_ridge_regression(std::shared_ptr<EchoStateNetwork> esn,
                                   const std::vector<std::vector<double>>& inputs,
                                   const std::vector<std::vector<double>>& targets,
                                   double regularization = 1e-6);
    
    void train_esn_online(std::shared_ptr<EchoStateNetwork> esn,
                         const std::vector<double>& input,
                         const std::vector<double>& target,
                         double learning_rate = 0.01);
    
    // Optimization methods
    std::shared_ptr<EchoStateNetwork> optimize_hyperparameters(
        const std::vector<std::vector<double>>& train_inputs,
        const std::vector<std::vector<double>>& train_targets,
        const std::vector<std::vector<double>>& val_inputs,
        const std::vector<std::vector<double>>& val_targets);
};

/**
 * ReservoirMetrics - Performance evaluation for reservoir systems
 */
class ReservoirMetrics
{
public:
    static double mean_squared_error(const std::vector<std::vector<double>>& predictions,
                                   const std::vector<std::vector<double>>& targets);
    
    static double normalized_root_mean_squared_error(const std::vector<std::vector<double>>& predictions,
                                                   const std::vector<std::vector<double>>& targets);
    
    static double memory_capacity(std::shared_ptr<ReservoirNode> reservoir,
                                size_t max_delay = 100);
    
    static double kernel_quality(std::shared_ptr<ReservoirNode> reservoir,
                               const std::vector<std::vector<double>>& inputs);
    
    // AtomSpace-specific metrics
    static Handle compute_performance_atoms(AtomSpacePtr as,
                                          const std::vector<double>& metrics);
};

/**
 * ReservoirEvolution - Evolutionary optimization for reservoir topologies
 */
class ReservoirEvolution
{
private:
    AtomSpacePtr _atomspace;
    size_t _population_size;
    double _mutation_rate;
    
public:
    ReservoirEvolution(AtomSpacePtr as, size_t pop_size = 50, double mut_rate = 0.1)
        : _atomspace(as), _population_size(pop_size), _mutation_rate(mut_rate) {}
    
    std::shared_ptr<EchoStateNetwork> evolve_topology(
        const std::vector<std::vector<double>>& train_inputs,
        const std::vector<std::vector<double>>& train_targets,
        size_t generations = 100);
    
    void mutate_weights(std::shared_ptr<ReservoirNode> reservoir, double strength = 0.1);
    void crossover_reservoirs(std::shared_ptr<ReservoirNode> parent1,
                            std::shared_ptr<ReservoirNode> parent2,
                            std::shared_ptr<ReservoirNode> offspring);
};

/**
 * ReservoirAnalysis - Advanced analysis tools for reservoir dynamics
 */
class ReservoirAnalysis
{
public:
    static std::vector<double> compute_lyapunov_exponents(std::shared_ptr<ReservoirNode> reservoir,
                                                        const std::vector<std::vector<double>>& inputs);
    
    static double compute_entropy_production(std::shared_ptr<ReservoirNode> reservoir,
                                           const std::vector<std::vector<double>>& inputs);
    
    static std::vector<std::vector<double>> extract_dynamics_matrix(std::shared_ptr<ReservoirNode> reservoir);
    
    // AtomSpace visualization and analysis
    static Handle create_dynamics_graph(AtomSpacePtr as, std::shared_ptr<ReservoirNode> reservoir);
    static void analyze_information_flow(AtomSpacePtr as, std::shared_ptr<ReservoirNode> reservoir);
};

} // namespace algorithms
} // namespace reservoir
} // namespace opencog

#endif // _OPENCOG_RESERVOIR_ALGORITHMS_H