#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"

#include <gtest/gtest.h>
#include <vector>

namespace{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynMat;
using cubeai::DynVec;
using namespace cubeai::rl::policies;

}

TEST(TestEpsilonGreedyPolicy, Test_1) {

    EpsilonGreedyPolicy policy(EpsilonGreedyPolicy::MIN_EPS);

    std::vector<real_t> vals{1.0, 2.0, 3.0};
    auto max_idx = policy(vals);

    ASSERT_EQ(policy.decay_option(), EpsilonDecayOption::NONE);
    ASSERT_EQ(max_idx, max_idx);

}

TEST(TestEpsilonGreedyPolicy, Test_2) {

    EpsilonGreedyPolicy policy(EpsilonGreedyPolicy::MIN_EPS,
                               42, EpsilonDecayOption::EXPONENTIAL,
                               EpsilonGreedyPolicy::MIN_EPS,
                               EpsilonGreedyPolicy::MAX_EPS,
                               EpsilonGreedyPolicy::EPSILON_DECAY_FACTOR);


    ASSERT_EQ(policy.decay_option(), EpsilonDecayOption::EXPONENTIAL);

}


TEST(TestEpsilonGreedyPolicy, Test_3) {

    EpsilonGreedyPolicy policy(EpsilonGreedyPolicy::MIN_EPS);

    DynVec<real_t> vals(3);
    vals[0] = 1.0;
    vals[1] = 2.0;
    vals[2] = 3.0;
    auto max_idx = policy(vals);

    ASSERT_EQ(max_idx, max_idx);

}

TEST(TestEpsilonGreedyPolicy, Test_4) {

    EpsilonGreedyPolicy policy(EpsilonGreedyPolicy::MIN_EPS);

    DynMat<real_t> vals(3, 3);
    vals(0,0) = 1.0;
    vals(0,1) = 2.0;
    vals(0,2) = 3.0;

    vals(1,0) = 1.0;
    vals(1,1) = 2.0;
    vals(1,2) = 3.0;

    vals(2,0) = 1.0;
    vals(2,1) = 2.0;
    vals(2,2) = 3.0;

    auto max_idx = policy(vals, 0);
    ASSERT_EQ(max_idx, max_idx);

}


