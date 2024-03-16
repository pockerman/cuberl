#include "cubeai/base/cubeai_types.h"
#include "cubeai/maths/vector_math.h"

#include <gtest/gtest.h>
#include <vector>

namespace{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynVec;


}


TEST(TestSoftMaxVector, Test_1) {

    DynVec<real_t> vals(4);
    for(uint_t i=0; i< static_cast<uint_t>(vals.size()); ++i){

        vals[i] = 10.0;
    }
    auto softmax_vec = cubeai::maths::softmax_vec(vals);

    EXPECT_DOUBLE_EQ(0.25, softmax_vec[0]);
    EXPECT_DOUBLE_EQ(0.25, softmax_vec[1]);
    EXPECT_DOUBLE_EQ(0.25, softmax_vec[2]);
    EXPECT_DOUBLE_EQ(0.25, softmax_vec[3]);

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
