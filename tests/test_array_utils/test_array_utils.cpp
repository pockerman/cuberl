#include "cubeai/utils/array_utils.h"

#include <gtest/gtest.h>
#include <vector>

namespace{

using cubeai::real_t;
using cubeai::uint_t;

}


TEST(TestArrayUtils, Test_bin_index) {

    std::vector<real_t> pos_bins{-1.2, -0.95714286, -0.71428571, -0.47142857, -0.22857143,  0.01428571,  0.25714286,  0.5 };
    auto pos = 0.43;
    auto bin_idx = cubeai::bin_index(pos, pos_bins);

    // bin index should be 7 according to
    // numpy equivalent function
    ASSERT_EQ(bin_idx, static_cast<uint_t>(7));


}
