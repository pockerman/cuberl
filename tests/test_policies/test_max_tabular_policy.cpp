#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/policies/max_tabular_policy.h"

#include <gtest/gtest.h>
#include <vector>

namespace{

using cubeai::real_t;
using cubeai::uint_t;
using namespace cubeai::rl::policies;

}


TEST(TestMaxTabularPolicy, Test_Constructor) {

    MaxTabularPolicy policy;
    std::vector<real_t> vals{1.0, 2.0, 3.0};
    policy(vals);

}

TEST(TestMaxTabularPolicy, Test_Operator_1) {

    MaxTabularPolicy policy;

    std::vector<real_t> vals{1.0, 2.0, 3.0};
    auto max_idx = policy(vals);

    ASSERT_EQ(max_idx, static_cast<uint_t>(2));

}

TEST(TestMaxTabularPolicy, Test_Operator_2) {

    MaxTabularPolicy policy;

    std::vector<real_t> vals{3.0, 2.0, 1.0};
    auto max_idx = policy(vals);

    ASSERT_EQ(max_idx, static_cast<uint_t>(0));

}

TEST(TestMaxTabularPolicy, Test_Operator_3) {

    MaxTabularPolicy policy;

    std::vector<real_t> vals{1.0, 3.0, 2.0};
    auto max_idx = policy(vals);

    ASSERT_EQ(max_idx, static_cast<uint_t>(1));

}
