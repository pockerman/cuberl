#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/iteration_counter.h"

#include <gtest/gtest.h>
#include <vector>

namespace{

using cubeai::real_t;
using cubeai::uint_t;

}


TEST(TestIterationCounter, Test_constructor) {

    cubeai::utils::IterationCounter counter(1);
    ASSERT_EQ(counter.max_iterations(), static_cast<uint_t>(1));
}

TEST(TestIterationCounter, Test_continue_iterations) {

    std::vector<real_t> vals{0.0, 1.0, 0.0};

    cubeai::utils::IterationCounter counter(1);
    ASSERT_EQ(counter.max_iterations(), static_cast<uint_t>(1));
    ASSERT_TRUE(counter.continue_iterations());
    ASSERT_FALSE(counter.continue_iterations());

}

