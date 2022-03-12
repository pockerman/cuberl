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
    KDTree<2, std::vector<real_t>> tree;
    ASSERT_TRUE(tree.empty());

    // attempt to insert the root
    std::vector<real_t> point(3, 1.0);
    auto iterator = tree.insert(point);

    ASSERT_TRUE(iterator != nullptr);
    ASSERT_EQ(tree.size(), static_cast<uint_t>(1));

}


TEST(TestKDTree, Test_search_1){

    // create an empty tree
    KDTree<3, std::vector<uint_t>> tree;
    ASSERT_TRUE(tree.empty());

    // attempt to insert the root
    std::vector<uint_t> point(3, 1);
    auto iterator = tree.insert(point);

    ASSERT_TRUE(iterator != nullptr);
    ASSERT_EQ(tree.size(), static_cast<uint_t>(1));

    auto* iterator_result = tree.search(point, [&](const auto& v1, const auto& v2){
        if(v1.size() != v2.size()){
            return false;
        }
        return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]);
    });

    ASSERT_TRUE(iterator_result != nullptr);
    ASSERT_EQ(iterator_result->level, static_cast<uint_t>(0));

}
