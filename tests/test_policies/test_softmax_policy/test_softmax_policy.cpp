#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/policies/softmax_policy.h"

#include <gtest/gtest.h>
#include <vector>

namespace{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynMat;
using cubeai::DynVec;
using namespace cubeai::rl::policies;

}


TEST(TestMaxTabularSoftmaxPolicy, Test_1) {

    MaxTabularSoftmaxPolicy policy;
    std::vector<real_t> vals{3.0, 2.0, 1.0 ,0.0};
    auto action = policy(vals);
    ASSERT_EQ(static_cast<uint_t>(0), action);

}

/*TEST(TestMaxTabularPolicy, Test_Operator_1) {

    RandomTabularPolicy policy(42);

    std::vector<real_t> vals{1.0, 2.0, 3.0};
    auto max_idx = policy(vals);

    ASSERT_EQ(max_idx, max_idx);

}


TEST(TestMaxTabularPolicy, Test_Operator_2) {

    RandomTabularPolicy policy(42);

    DynVec<real_t> vals(3);
    vals[0] = 1.0;
    vals[1] = 2.0;
    vals[2] = 3.0;
    auto max_idx = policy(vals);

    ASSERT_EQ(max_idx, max_idx);

}

TEST(TestMaxTabularPolicy, Test_Operator_3) {

    RandomTabularPolicy policy(42);

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

}*/
