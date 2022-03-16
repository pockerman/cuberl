#ifndef KD_TREE_H
#define KD_TREE_H

/**
 * Basic implementation of KD-Tree data structure.
 * The implementation herein follows the implementation
 * in the book Advanced Algorithsm and Data Structures by
 * Manning publications
 */

#include "cubeai/base/cubeai_consts.h"
#include "cubeai/base/cubeai_config.h"

#ifdef CUBEAI_DEBUG
#include <cassert>
#endif

#include <memory>

namespace cubeai {
namespace contaiers {

///
/// \brief
///
template<typename DataType>
struct KDTreeNode
{

    typedef DataType data_type;
    typedef typename DataType::value_type value_type;

    data_type data;
    std::shared_ptr<KDTreeNode<data_type>> left;
    std::shared_ptr<KDTreeNode<data_type>> right;
    uint_t level;
    uint_t n_copies{0};

    ///
    ///
    ///
    KDTreeNode()=default;

    ///
    /// \brief KDTreeNode
    /// \param level
    /// \param left_
    /// \param right_
    ///
    KDTreeNode(uint_t level, const data_type& data_,
               std::shared_ptr<KDTreeNode<data_type>> left_,  std::shared_ptr<KDTreeNode<data_type>> right_);

    ///
    ///
    ///
    value_type get_key(uint_t i)const noexcept{return data[i];}
};

template<typename DataType>
KDTreeNode<DataType>::KDTreeNode(uint_t level_, const data_type& data_,  std::shared_ptr<KDTreeNode<data_type>> left_,
           std::shared_ptr<KDTreeNode<data_type>> right_)
    :
    data(data_),
    left(left_),
    right(right_),
    level(level_),
    n_copies(1)
{}

namespace detail{

// helper funcions for KDTree
template<typename PointType>
typename PointType::value_type
get_point_key(const uint_t level, const uint_t k, const PointType& point ){

    // extract the point value at index level mod k
    // zero-based indexing is assumed
    auto j = level % k;
    return point[j];
}

///
///
///
template<typename DataType>
int compare(std::shared_ptr<KDTreeNode<DataType>> node, const DataType& data, const uint_t k){

    auto node_key = get_point_key(node->level, k, node->data);
    auto point_key = get_point_key(node->level, k, data);

    if (point_key - node_key < 0){
        return -1;
    }
    else if(point_key - node_key > 0){
        return 1;
    }

    return 0;

}


}




///
/// \detailed The KDTree class. It models
/// a k-d tree where k is the dimension of the space where the DataType
/// belongs to. Specifically, A k-d tree is a binary search tree whose elements are points taken
/// from a k-dimensional space, that is, tuples with k elements, whose coordinates can be compared.
///  In addition to that, in a k-d tree, at each level i, we only compare the i-th (modulo k ) coordinate of points to
/// decide which branch of the tree will be traversed.
///
/// Invariants
///
/// - All points in the tree have dimension k
/// - Each level has a split coordinate index j , such that 0 ≤ j < k .
/// - If a node N ’s split coordinate index is j , then N ’s children have a split coordinate
/// equal to (j+1) mod k
/// - For each node N , with split coordinate index j , all nodes L in its left subtree have
/// a smaller value for N ’s split coordinate, L[j] < N[j] , and all nodes R on N ’s right
/// subtree have a larger or equal value for N ’s split coordinate, R[j] ≥ N[j] .
///
template<int k, typename DataType>
class KDTree
{
public:

    typedef DataType data_type;
    typedef KDTreeNode<DataType> node_type;

    typedef node_type* iterator;
    typedef const node_type* const_iterator;

    ///
    /// \brief KDTree
    ///
    KDTree() = default;

    ///
    /// \brief empty
    /// \return
    ///
    bool empty()const noexcept{return root_ == nullptr;}

    ///
    /// \brief size
    /// \return
    ///
    uint_t size()const noexcept{return n_nodes_;}

    ///
    /// \brief search Search for the data in the tree
    /// \param data
    /// \return
    ///
    template<typename ComparisonPolicy>
    const_iterator search(const data_type& data, const ComparisonPolicy& calculator)const{ return search_(root_, data, calculator);}

    ///
    /// \brief insert
    /// \param data
    /// \return
    ///
    template<typename ComparisonPolicy>
    iterator insert(const data_type& data, const ComparisonPolicy& comparison_policy);

private:


    ///
    /// \brief root_ The root of the tree
    ///
    std::shared_ptr<node_type> root_;

    ///
    ///
    ///
    uint_t n_nodes_{0};

    ///
    /// \brief search_. Recursion-based adapter to perform tree-search.
    ///
    template<typename ComparisonPolicy>
    const_iterator search_(std::shared_ptr<node_type> node, const data_type& data,
                           const ComparisonPolicy& calculator)const;

    ///
    /// \brief insert_ Recursion-based adapter to perform insertion
    /// in the KDTree
    ///
    template<typename ComparisonPolicy>
    iterator insert_(std::shared_ptr<node_type>& node, const data_type& data,
                     const ComparisonPolicy& calculator, uint_t level);

};

template<int k, typename DataType>
template<typename ComparisonPolicy>
typename KDTree<k, DataType>::iterator
KDTree<k, DataType>::insert(const data_type& data, const ComparisonPolicy& comparison_policy){

#ifdef CUBEAI_DEBUG
    assert(data.size() == k && "Data size not equal to k.");
#endif

    // if the root node is null then
    // simply insert
    if(!root_){
        root_ = std::make_shared<node_type>(0, data, nullptr, nullptr);
        n_nodes_++;
        return root_.get();
    }

    return insert_(root_, data, comparison_policy, 0);
}

template<int k, typename DataType>
template<typename DistanceCalculator>
typename KDTree<k, DataType>::const_iterator
KDTree<k, DataType>::search_(std::shared_ptr<node_type> node, const data_type& data, const DistanceCalculator& calculator)const{

    if(!node){
        return nullptr;
    }

    if(calculator(node->data, data)){
        return node.get();
    }
    else if(detail::compare(node, data, k) < 0){
        return search_(node->left, data, calculator);
    }
    else{
        return search_(node->right, data, calculator);
    }
}

template<int k, typename DataType>
template<typename ComparisonPolicy>
typename KDTree<k, DataType>::iterator
KDTree<k, DataType>::insert_(std::shared_ptr<node_type>& node, const data_type& data,
                             const ComparisonPolicy& calculator, uint_t level){

    node_type* node_ptr;

    if(!node){
        node = std::make_shared<node_type>(level, data, nullptr, nullptr);
        node_ptr = node.get();
        n_nodes_++;
    }
    else{

        if(calculator(node->data, data)){

            // we found the data increase the counter
            node->n_copies += 1;
            return node.get();
        }
        else if(detail::compare(node, data, k) < 0){
            node_ptr = insert_(node->left, data, calculator, node->level + 1);
        }
        else{

            node_ptr = insert_(node->right, data, calculator, node->level + 1);
        }
    }

    return node_ptr;
}




}

}

#endif // KD_TREE_H
