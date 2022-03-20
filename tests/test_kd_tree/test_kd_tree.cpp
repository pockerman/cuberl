#include "cubeai/data_structs/kd_tree.h"
#include "cubeai/base/cubeai_types.h"

#include <gtest/gtest.h>
#include <vector>

namespace{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynVec;
using cubeai::DynMat;
using cubeai::contaiers::KDTree;
using cubeai::contaiers::KDTreeNode;

typedef KDTreeNode<std::vector<real_t>> node_type;
}

TEST(TestKDTree, Test_default_constructor){

    KDTree<node_type> tree(3);
    ASSERT_TRUE(tree.empty());

}


TEST(TestKDTree, Test_insert_root) {

    // create an empty tree
    KDTree<node_type> tree(2);
    ASSERT_TRUE(tree.empty());

    auto criterion = [&](const auto& v1, const auto& v2){
        if(v1.size() != v2.size()){
            return false;
        }
        return (v1[0] == v2[0] && v1[1] == v2[1]);
    };

    // attempt to insert the root
    std::vector<real_t> point(2, 1.0);
    auto iterator = tree.insert(point, criterion);

    ASSERT_TRUE(iterator != nullptr);
    ASSERT_EQ(tree.size(), static_cast<uint_t>(1));

}

TEST(TestKDTree, Test_insert_root_2) {

    // create an empty tree
    KDTree<node_type> tree(2);
    ASSERT_TRUE(tree.empty());

    auto criterion = [&](const auto& v1, const auto& v2){
        if(v1.size() != v2.size()){
            return false;
        }
        return (v1[0] == v2[0] && v1[1] == v2[1]);
    };

    // attempt to insert the root
    std::vector<real_t> point(2, 1.0);
    auto iterator = tree.insert(point, criterion);
    ASSERT_EQ(tree.size(), static_cast<uint_t>(1));

    // reinsert again
    iterator = tree.insert(point, criterion);

    ASSERT_TRUE(iterator != nullptr);

    // the size remains the same as we simply
    // increase the multitude
    ASSERT_EQ(tree.size(), static_cast<uint_t>(1));
    ASSERT_EQ(iterator->n_copies, static_cast<uint_t>(2));

}


TEST(TestKDTree, Test_search_1){

    // create an empty tree
    KDTree<node_type> tree(3);
    ASSERT_TRUE(tree.empty());

    auto criterion = [&](const auto& v1, const auto& v2){
        if(v1.size() != v2.size()){
            return false;
        }
        return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]);
    };

    // attempt to insert the root
    std::vector<real_t> point(3, 1);
    auto iterator = tree.insert(point, criterion);

    ASSERT_TRUE(iterator != nullptr);
    ASSERT_EQ(tree.size(), static_cast<uint_t>(1));

    auto iterator_result = tree.search(point, criterion);

    ASSERT_TRUE(iterator_result != nullptr);
    ASSERT_EQ(iterator_result->level, static_cast<uint_t>(0));

}


TEST(TestKDTree, Test_range_constructor){

    typedef KDTreeNode<std::vector<uint_t>> node_type;

    std::vector<std::vector<uint_t>> mat(4);

    for(uint_t i=0; i<mat.size(); ++i){
        mat[i].resize(4);
        for(uint_t j=0; j<mat[i].size(); ++j){

            if( i != j){
                mat[i ][ j ] = 0;
            }
            else{
               mat[i ][ j ]= 1;
            }
        }

    }

    auto criterion = [&](const auto& v1, const auto& v2){
        if(v1.size() != v2.size()){
            return false;
        }
        return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2] && v1[3] == v2[3]);
    };

    // create an empty tree
    KDTree<node_type> tree(4, mat.begin(), mat.end(), criterion);
    ASSERT_FALSE(tree.empty());



    ASSERT_EQ(tree.size(), static_cast<uint_t>(4));

    /*auto* iterator_result = tree.search(point, criterion);

    ASSERT_TRUE(iterator_result != nullptr);
    ASSERT_EQ(iterator_result->level, static_cast<uint_t>(0));*/

}
