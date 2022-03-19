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
#include "cubeai/data_structs/fixed_size_priority_queue.h"

#ifdef CUBEAI_DEBUG
#include <cassert>
#endif


#include <vector>
#include <tuple>
#include <memory>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iterator>
#include <utility>

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
    /// \brief Constructor
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
    /// \brief get_key. Returns the key that corresponds to the i-th coordinate
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
int
compare(std::shared_ptr<KDTreeNode<DataType>> node, const DataType& data, const uint_t k){

    auto node_key = get_point_key(node->level, k, node->data);
    auto point_key = get_point_key(node->level, k, data);

    auto sign = point_key - node_key;

    if (sign < 0){
        return -1;
    }
    else if(sign > 0){
        return 1;
    }

    /// If this value is 0, meaning that the value of
    /// the coordinate compared is the same, then
    /// we go left half of the time, and right the other
    /// half. Otherwise, we just return the sign that
    /// will be either 1 or -1.
    return node->level % 2 == 0 ? -1 : 1;
}

///
/// Computes the distance between a point
/// and its projection on the split line passing
/// through a node. This distance is nothing other than the
/// absolute value of the difference between
/// the j -th coordinates of the two points,
/// where j = node.level mod k
///
template<typename DataType>
typename DataType::value_type
split_distance(std::shared_ptr<KDTreeNode<DataType>> node, const DataType& data, const uint_t k){

    auto node_key = get_point_key(node->level, k, node->data);
    auto point_key = get_point_key(node->level, k, data);
    return std::abs(node_key - point_key);
}

///
/// Partition the range specified by the given iterators
/// into left-median-right. The coordinate chosen depends
/// on the level passed.

template<typename Iterator>
std::tuple<typename std::iterator_traits<Iterator>::value_type,
           std::pair<Iterator, Iterator>,
           std::pair<Iterator, Iterator>>
partiion_on_median(Iterator begin, Iterator end, uint_t level, uint_t k){

    auto idx = get_point_key(leve, )

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
template<typename DataType>
class KDTree
{
public:

    typedef DataType data_type;
    typedef KDTreeNode<DataType> node_type;

    typedef node_type* iterator;
    typedef const node_type* const_iterator;

    ///
    /// \brief KDTree. Constructor
    ///
    KDTree(uint_t k);

    ///
    ///
    ///
    template<typename Iterator>
    KDTree(uint_t k, Iterator begin, Iterator end);

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

    ///
    /// \brief nearest_search. Returns an ordered vector of the n closest
    /// data points to the given target data.
    ///
    template<typename ComparisonPolicy>
    std::vector<data_type> nearest_search(const data_type& data, uint_t n, const ComparisonPolicy& calculator)const
    {return nearest_search_(root_, data, n, calculator);}

private:


    ///
    /// \brief root_. The root of the tree
    ///
    std::shared_ptr<node_type> root_;

    ///
    /// \brief n_nodes_. Number of nodes in the tree
    ///
    uint_t n_nodes_{0};

    ///
    /// \brief k_ The spatial dimension of the data
    ///
    uint_t k_;

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

    ///
    /// \brief nearest_search_ Adapter to perform nearest search
    ///
    template<typename ComparisonPolicy>
    std::vector<data_type> nearest_search_(std::shared_ptr<node_type>& node, const data_type& data,
                                           uint_t n, const ComparisonPolicy& calculator)const;

    ///
    /// \brief nearest_search_. Recursion-based adapter to perform nearest serach
    ///
    template<typename ComparisonPolicy, typename PriorityQueueType>
    void nearest_search_(std::shared_ptr<node_type> node, const data_type& data,
                                           const ComparisonPolicy& calculator, PriorityQueueType& pq)const;


    ///
    ///
    ///
    template<typename Iterator>
    void assign_(Iterator begin, Iterator end){create_(begin, end, 0);}

    template<typename Iterator>
    void create_(Iterator begin, Iterator end, uint_t level);


    template<typename Iterator>
    std::shared_ptr<node_type> do_create_(Iterator begin, Iterator end, uint_t level);

};

template<typename DataType>
KDTree<DataType>::KDTree(uint_t k)
    :
    root_(),
    n_nodes_(0),
    k_(k)
{}

template<typename DataType>
template<typename Iterator>
KDTree<DataType>::KDTree(uint_t k, Iterator begin, Iterator end)
    :
     KDTree<DataType>(k)
{
    assign(begin, end);
}

template<typename DataType>
template<typename ComparisonPolicy>
typename KDTree<DataType>::iterator
KDTree<DataType>::insert(const data_type& data, const ComparisonPolicy& comparison_policy){

#ifdef CUBEAI_DEBUG
    assert(data.size() == k_ && "Data size not equal to k.");
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

template<typename DataType>
template<typename ComparisonPolicy>
std::vector<typename KDTree<DataType>::data_type>
KDTree<DataType>::nearest_search_(std::shared_ptr<node_type>& node, const data_type& data,
                                     uint_t n, const ComparisonPolicy& calculator)const{

    typedef std::pair<typename ComparisonPolicy::value_type, std::shared_ptr<node_type>> value_type;

    struct comparison
    {
        typedef std::greater<typename ComparisonPolicy::value_type> comparison_type;
        bool operator()(const value_type& v1, const value_type& v2)const{
            return comparison_type(v1.first, v2.first);
        }
    };

    // initialize a min-heap
    //std::priority_queue<value_type, decltype(compare)> pq;
    cubeai::containers::FixedSizeMinPriorityQueue<value_type, comparison, std::vector<value_type>> pq(n);

    /// Before starting our search, we need to initialize
    /// the priority queue by adding a “guard”: a tuple
    /// containing infinity as distance, a value that will be
    /// larger than any other distance computed, and so
    /// it will be the first tuple removed from the queue
    /// if we find at least n points in the tree
    pq.push(std::make_pair(std::numeric_limits<typename ComparisonPolicy::value_type>::max(),std::shared_ptr<node_type>()));

    nearest_search_(node, data, calculator, pq);

    auto peek = pq.top();

    /// ... if its top element is still at an infinite
    /// distance, we need to remove it, because it
    /// means we have added less than n elements
    /// to the queue

    if(!peek.second){
        peek.pop();
    }

    std::vector<typename ComparisonPolicy::value_type, DataType> result;
    result.reserve(pq.size());
    while(!pq.empty()){
        auto item = pq.top_and_pop();
        result.push_back(item);
    }

    // loop over the priority queue and collect
    // the points we found

    return result;
}

template<typename DataType>
template<typename ComparisonPolicy, typename PriorityQueueType>
void
KDTree<DataType>::nearest_search_(std::shared_ptr<node_type> node, const data_type& data,
                                     const ComparisonPolicy& calculator, PriorityQueueType& pq)const{


    if(!node){
        // nothing to do with the queue
        return;
    }
    else{

        // compute the distance between the target
        // and the data hold by the node
        auto dist = calculator.distance(node->data, data);

        // insrt into the queue
        pq.push(std::make_pair(dist, node));

        if(detail::compare(node, data, k_) < 0){

            auto close_branch = node->left;
            nearest_search_(close_branch, data, calculator, pq );

            auto peek = pq.top();

            if(split_distance(node, data) < peek.first){
                nearest_search_(node->right, data, calculator, pq );
            }
        }
        else{

            auto close_branch = node->right;
            nearest_search_(close_branch, data, calculator, pq );

            auto peek = pq.top();

            if(split_distance(node, data) < peek.first){
                nearest_search_(node->left, data, calculator, pq );
            }
        }
    }
}

template<typename DataType>
template<typename DistanceCalculator>
typename KDTree<DataType>::const_iterator
KDTree<DataType>::search_(std::shared_ptr<node_type> node, const data_type& data, const DistanceCalculator& calculator)const{

    if(!node){
        return nullptr;
    }

    if(calculator(node->data, data)){
        return node.get();
    }
    else if(detail::compare(node, data, k_) < 0){
        return search_(node->left, data, calculator);
    }
    else{
        return search_(node->right, data, calculator);
    }
}

template<typename DataType>
template<typename ComparisonPolicy>
typename KDTree<DataType>::iterator
KDTree<DataType>::insert_(std::shared_ptr<node_type>& node, const data_type& data,
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
        else if(detail::compare(node, data, k_) < 0){
            node_ptr = insert_(node->left, data, calculator, node->level + 1);
        }
        else{

            node_ptr = insert_(node->right, data, calculator, node->level + 1);
        }
    }

    return node_ptr;
}

template<typename DataType>
template<typename Iterator>
void
KDTree<DataType>::create_(Iterator begin, Iterator end, uint_t level){



    auto n_points = std::distance(begin, end);

    // nothing to do if no points
    // are given
    if(n_points == 0){
        return ;
    }

    if(n_points == 1){
        auto data = *begin;

        // create the root
        root_ = std::make_shared<typename KDTree<DataType>::node_type>(level, data, nullptr, nullptr);
    }

    // otherwise partition the range
    auto [median, left, right] = partiion(begin, end, level);

    // create root
    root_ = std::make_shared<typename KDTree<DataType>::node_type>(level, median, nullptr, nullptr);

    // create left and right subtrees
    auto left_tree = do_create_(left.begin(), left.end(), level + 1);

    // create left and right subtrees
    auto right_tree = do_create_(right.begin(), right.end(), level + 1);

    root_->left = left;
    root_->right = right;
}

template<typename DataType>
template<typename Iterator>
std::shared_ptr<typename KDTree<DataType>::node_type>
            KDTree<DataType>::do_create_(Iterator begin, Iterator end, uint_t level){

    auto n_points = std::distance(begin, end);

    // nothing to do if no points
    // are given
    if(n_points == 0){
        return nullptr;
    }

    if(n_points == 1){
        auto data = *begin;
        return std::make_shared<typename KDTree<DataType>::node_type>(level, data, nullptr, nullptr);
    }

    // otherwise partition the range
    auto [median, left, right] = partiion(begin, end, level);

    auto left_tree = do_create_(left.begin(), left.end(), level + 1);
    auto right_tree = do_create_(right.begin(), right.end(), level + 1);
    return std::make_shared<typename KDTree<DataType>::node_type>(level, median, left, right);

}


}

}

#endif // KD_TREE_H
