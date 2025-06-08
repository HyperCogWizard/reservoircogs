/*
 * basic_esn_atomspace.cpp
 * 
 * Basic example of Echo State Network with AtomSpace integration
 * This demonstrates the core functionality of ReservoirCogs C++ API
 */

#include <iostream>
#include <vector>
#include <random>
#include <opencog/atomspace/AtomSpace.h>
#include <opencog/reservoir/nodes/ReservoirNode.h>
#include <opencog/reservoir/algorithms/ReservoirAlgorithms.h>

using namespace opencog;
using namespace opencog::reservoir;
using namespace opencog::reservoir::algorithms;

int main() {
    std::cout << "ReservoirCogs Basic ESN with AtomSpace Example\n";
    std::cout << "==============================================\n\n";
    
    // Create AtomSpace for symbolic AI integration
    auto atomspace = createAtomSpace();
    std::cout << "Created AtomSpace for symbolic knowledge storage.\n";
    
    // Create Echo State Network
    size_t reservoir_size = 100;
    size_t input_size = 3;
    size_t output_size = 1;
    
    auto esn = std::make_shared<EchoStateNetwork>(
        atomspace, reservoir_size, input_size, output_size);
    std::cout << "Created ESN with " << reservoir_size << " reservoir units.\n";
    
    // Configure reservoir parameters
    esn->set_leaking_rate(0.3);
    esn->set_spectral_radius(0.9);
    std::cout << "Configured ESN parameters (leaking_rate=0.3, spectral_radius=0.9).\n";
    
    // Generate sample training data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);
    
    size_t n_samples = 1000;
    std::vector<std::vector<double>> train_inputs(n_samples);
    std::vector<std::vector<double>> train_targets(n_samples);
    
    for (size_t i = 0; i < n_samples; ++i) {
        train_inputs[i].resize(input_size);
        train_targets[i].resize(output_size);
        
        for (size_t j = 0; j < input_size; ++j) {
            train_inputs[i][j] = dis(gen);
        }
        
        // Simple target function: sum of inputs with nonlinearity
        double sum = 0.0;
        for (double x : train_inputs[i]) {
            sum += x;
        }
        train_targets[i][0] = std::tanh(sum);
    }
    
    std::cout << "Generated " << n_samples << " training samples.\n";
    
    // Train the ESN using ridge regression
    ReservoirTrainer trainer(atomspace);
    std::cout << "Training ESN with ridge regression...\n";
    
    trainer.train_esn_ridge_regression(esn, train_inputs, train_targets, 1e-6);
    std::cout << "Training completed.\n";
    
    // Store trained model in AtomSpace
    esn->store_weights_to_atomspace();
    std::cout << "Stored trained weights in AtomSpace.\n";
    
    // Test the trained model
    std::vector<std::vector<double>> test_inputs(100);
    std::vector<std::vector<double>> predictions(100);
    
    for (size_t i = 0; i < 100; ++i) {
        test_inputs[i].resize(input_size);
        for (size_t j = 0; j < input_size; ++j) {
            test_inputs[i][j] = dis(gen);
        }
        predictions[i] = esn->predict(test_inputs[i]);
    }
    
    std::cout << "Generated predictions for 100 test samples.\n";
    
    // Compute performance metrics
    std::vector<std::vector<double>> test_targets(100);
    for (size_t i = 0; i < 100; ++i) {
        test_targets[i].resize(output_size);
        double sum = 0.0;
        for (double x : test_inputs[i]) {
            sum += x;
        }
        test_targets[i][0] = std::tanh(sum);
    }
    
    double mse = ReservoirMetrics::mean_squared_error(predictions, test_targets);
    double nrmse = ReservoirMetrics::normalized_root_mean_squared_error(predictions, test_targets);
    double memory_cap = ReservoirMetrics::memory_capacity(esn, 50);
    
    std::cout << "\nPerformance Metrics:\n";
    std::cout << "Mean Squared Error: " << mse << "\n";
    std::cout << "Normalized RMSE: " << nrmse << "\n";
    std::cout << "Memory Capacity: " << memory_cap << "\n";
    
    // Store metrics in AtomSpace
    std::vector<double> metrics = {mse, nrmse, memory_cap};
    Handle perf_handle = ReservoirMetrics::compute_performance_atoms(atomspace, metrics);
    std::cout << "\nStored performance metrics in AtomSpace.\n";
    
    // Analyze reservoir dynamics
    std::cout << "\nAnalyzing reservoir dynamics...\n";
    auto lyapunov_exp = ReservoirAnalysis::compute_lyapunov_exponents(esn, test_inputs);
    std::cout << "Computed Lyapunov exponents for dynamics analysis.\n";
    
    Handle dynamics_graph = ReservoirAnalysis::create_dynamics_graph(atomspace, esn);
    std::cout << "Created dynamics graph in AtomSpace.\n";
    
    // Display AtomSpace statistics
    std::cout << "\nAtomSpace Statistics:\n";
    std::cout << "Total atoms: " << atomspace->get_size() << "\n";
    std::cout << "Nodes: " << atomspace->get_num_nodes() << "\n";
    std::cout << "Links: " << atomspace->get_num_links() << "\n";
    
    std::cout << "\nExample completed successfully!\n";
    std::cout << "The reservoir has been trained and integrated with AtomSpace.\n";
    std::cout << "All symbolic knowledge is now available for reasoning and analysis.\n";
    
    return 0;
}