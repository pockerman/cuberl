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

namespace detail{

// helper funcions for KDTree
template<typename PointType>
typename PointType::value_type
get_node_key_(const uint_t level, const uint_t k, const PointType& point ){

    // extract the point value at index level mod k
    // zero-based indexing is assumed
    auto j = level % k;
    return point[j];
}




}

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
    template<typename DistanceCalculator>
    const_iterator search(const data_type& data, const DistanceCalculator& calculator)const{ return search_(root_, data, calculator);}

    ///
    /// \brief insert
    /// \param data
    /// \return
    ///
    iterator insert(const data_type& data);

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
    /// \brief search_. Adapter to perform the serach
    ///
    template<typename DistanceCalculator>
    const_iterator search_(std::shared_ptr<node_type> node, const data_type& data, const DistanceCalculator& calculator)const;

};

template<int k, typename DataType>
typename KDTree<k, DataType>::iterator
KDTree<k, DataType>::insert(const data_type& data){

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

    return nullptr;
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
}


}

}

#endif // KD_TREE_H
