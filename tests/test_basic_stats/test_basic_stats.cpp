#include "cubeai/base/cubeai_types.h"
#include "cubeai/maths/basic_array_statistics.h"

#include <gtest/gtest.h>
#include <vector>

namespace{

using cubeai::real_t;
using cubeai::uint_t;

}


TEST(TestArrayUtils, Test_mean) {

    std::vector<uint_t> vals{1, 1};

    auto mean = cubeai::maths::stats::mean(vals);
    EXPECT_DOUBLE_EQ(mean, 1);

}

TEST(TestArrayUtils, Test_choice) {

    std::vector<real_t> vals{0.0, 1.0, 0.0};

    auto idx = cubeai::maths::stats::choice(vals);
    ASSERT_EQ(idx, static_cast<uint_t>(1));

}


TEST(TestArrayUtils, Test_choice_2) {

    std::vector<real_t> probs{0.0, 1.0, 0.0};
    std::vector<uint_t> choices{1, 2, 3};

    auto idx = cubeai::maths::stats::choice(choices, probs);
    ASSERT_EQ(idx, static_cast<uint_t>(2));

}
