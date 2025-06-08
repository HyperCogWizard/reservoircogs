/*
 * test_reservoir_node.cc
 * 
 * Unit tests for ReservoirNode and EchoStateNetwork
 */

#include <gtest/gtest.h>
#include "../opencog/reservoir/nodes/ReservoirNode.h"
#include <opencog/atomspace/AtomSpace.h>

using namespace opencog;
using namespace opencog::reservoir;

class ReservoirNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        as = createAtomSpace();
    }
    
    AtomSpacePtr as;
};

TEST_F(ReservoirNodeTest, BasicConstruction) {
    auto reservoir = std::make_shared<ReservoirNode>(as, 100);
    EXPECT_EQ(reservoir->size(), 100);
    EXPECT_TRUE(reservoir->get_handle() != nullptr);
}

TEST_F(ReservoirNodeTest, StateUpdate) {
    auto reservoir = std::make_shared<ReservoirNode>(as, 10);
    std::vector<double> input = {1.0, 0.5, -0.3};
    
    auto state = reservoir->update(input);
    EXPECT_EQ(state.size(), 10);
    
    // State should be different from initial zero state
    bool non_zero = false;
    for (double val : state) {
        if (std::abs(val) > 1e-6) {
            non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(non_zero);
}

TEST_F(ReservoirNodeTest, StateReset) {
    auto reservoir = std::make_shared<ReservoirNode>(as, 10);
    std::vector<double> input = {1.0, 0.5, -0.3};
    
    // Update to change state
    reservoir->update(input);
    
    // Reset and verify all zeros
    reservoir->reset_state();
    auto state = reservoir->get_state();
    
    for (double val : state) {
        EXPECT_NEAR(val, 0.0, 1e-10);
    }
}

class EchoStateNetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        as = createAtomSpace();
    }
    
    AtomSpacePtr as;
};

TEST_F(EchoStateNetworkTest, BasicConstruction) {
    auto esn = std::make_shared<EchoStateNetwork>(as, 100, 3, 2);
    EXPECT_EQ(esn->size(), 100);
}

TEST_F(EchoStateNetworkTest, PredictionOutput) {
    auto esn = std::make_shared<EchoStateNetwork>(as, 50, 2, 1);
    std::vector<double> input = {0.5, -0.3};
    
    auto output = esn->predict(input);
    EXPECT_EQ(output.size(), 1);
}

TEST_F(EchoStateNetworkTest, LeakingRateEffect) {
    auto esn1 = std::make_shared<EchoStateNetwork>(as, 50, 2, 1);
    auto esn2 = std::make_shared<EchoStateNetwork>(as, 50, 2, 1);
    
    esn1->set_leaking_rate(0.1);
    esn2->set_leaking_rate(1.0);
    
    std::vector<double> input = {1.0, 0.0};
    
    auto state1 = esn1->update(input);
    auto state2 = esn2->update(input);
    
    // States should be different due to different leaking rates
    bool different = false;
    for (size_t i = 0; i < state1.size(); ++i) {
        if (std::abs(state1[i] - state2[i]) > 1e-6) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}