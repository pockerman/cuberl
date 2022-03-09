#include "cubeai/data_structs/kd_tree.h"
#include "cubeai/base/cubeai_types.h"

#include <gtest/gtest.h>
#include <vector>

namespace{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::contaiers::KDTree;

}

TEST(TestKDTree, Test_default_constructor){

    KDTree<3, std::vector<real_t>> tree;
    ASSERT_TRUE(tree.empty());

}


TEST(TestKDTree, Test_insert_root) {

    // create an empty tree
    KDTree<3, std::vector<real_t>> tree;
    ASSERT_TRUE(tree.empty());

    // attempt to insert the root
    std::vector<real_t> point(3, 1.0);
    auto iterator = tree.insert(point);

    ASSERT_TRUE(iterator != nullptr);
    ASSERT_EQ(tree.size(), static_cast<uint_t>(1));

}
