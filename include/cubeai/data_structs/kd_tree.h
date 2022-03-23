#ifndef KD_TREE_H
#define KD_TREE_H

/**
 * Basic implementation of KD-Tree data structure.
 * The implementation herein follows the implementation
 * in the book Advanced Algorithsm and Data Structures by
 * Manning publications
 */

#include "cubeai/base/cubeai_consts.h"
#include "cubeai/utils/cubeai_traits.h"
//#include "cubeai/base/cubeai_config.h"
#include "cubeai/data_structs/fixed_size_priority_queue.h"

/*
#ifdef CUBEAI_DEBUG
#include <cassert>
#endif
*/


#include <vector>
#include <tuple>
#include <memory>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iterator>
#include <utility>

namespace cubeai {
namespace containers {

///
/// \brief Default structure to represent
/// a node in a KDTree
///
template<typename DataType>
struct KDTreeNode
{

    typedef DataType data_type;
    typedef typename utils::vector_value_type_trait<DataType>::value_type value_type;

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

    ///
    /// \brief get_point_key
    /// \param level
    /// \param k
    /// \param point
    /// \return
    ///
    value_type
    get_point_key(const uint_t level, const uint_t k, const data_type& point ){

        // extract the point value at index level mod k
        // zero-based indexing is assumed
        auto j = level % k;
        return point[j];
    }
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

///
///
///
template<typename Node, typename DataType>
int
compare(std::shared_ptr<Node> node, const DataType& data, const uint_t k){

    auto node_key = node->get_point_key(node->level, k, node->data);
    auto point_key = node->get_point_key(node->level, k, data);

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
template<typename Node, typename DataType>
typename utils::vector_value_type_trait<DataType>::value_type
split_distance(std::shared_ptr<Node> node, const DataType& data, const uint_t k){

    auto node_key = node->get_point_key(node->level, k, node->data);
    auto point_key = node->get_point_key(node->level, k, data);
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
partiion_on_median(Iterator&& begin, Iterator&& end, uint_t level, uint_t k){


    typedef typename std::iterator_traits<Iterator>::value_type value_type;

    auto begin_ = begin;
    auto end_ = end;

    auto n_points = std::distance(begin_, end_);

    // the median index
    auto median_idx = n_points % 2 == 0 ? (n_points + 1) / 2 : n_points / 2;

    auto compare = [&](const value_type& v1, const value_type& v2){

        auto idx = level % k;
        return v1[idx] < v2[idx];
    };

    // rearrange the elements
    std::nth_element(begin_, begin_ + median_idx, end_ , compare);

    auto median = *(begin_ + median_idx);

    auto left = std::make_pair<Iterator,   Iterator>(std::forward<Iterator>(begin_), begin_ + median_idx);
    auto right = std::make_pair<Iterator,   Iterator>(begin_ + median_idx + 1, std::forward<Iterator>(end_));
    return std::make_tuple(median, left, right);

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
template<typename NodeType>
class KDTree
{
public:

    typedef NodeType node_type;
    typedef typename node_type::data_type data_type;

    typedef node_type* iterator;
    typedef const node_type* const_iterator;

    ///
    /// \brief KDTree. Constructor
    ///
    KDTree(uint_t k);

    ///
    ///
    ///
    template<typename Iterator, typename ComparisonPolicy>
    KDTree(uint_t k, Iterator begin, Iterator end, const ComparisonPolicy& policy);

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
    std::shared_ptr<node_type>
    search(const data_type& data, const ComparisonPolicy& calculator)const{ return search_(root_, data, calculator);}

    ///
    /// \brief insert
    /// \param data
    /// \return
    ///
    template<typename ComparisonPolicy>
    std::shared_ptr<node_type>
    insert(const data_type& data, const ComparisonPolicy& comparison_policy);

    ///
    /// \brief nearest_search. Returns an ordered vector of the n closest
    /// data points to the given target data.
    ///
    template<typename ComparisonPolicy>
    std::vector<std::pair<typename ComparisonPolicy::value_type, typename NodeType::data_type>>
    nearest_search(const data_type& data, uint_t n, const ComparisonPolicy& calculator)const
    {return nearest_search_(root_, data, n, calculator);}

    ///
    /// \brief dim
    /// \return
    ///
    uint_t dim()const noexcept{return k_;}

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
    std::shared_ptr<node_type> search_(std::shared_ptr<node_type> node, const data_type& data,
                           const ComparisonPolicy& calculator)const;

    ///
    /// \brief insert_ Recursion-based adapter to perform insertion
    /// in the KDTree
    ///
    template<typename ComparisonPolicy>
    std::shared_ptr<node_type> insert_(std::shared_ptr<node_type>& node, const data_type& data,
                                      const ComparisonPolicy& calculator, uint_t level);

    ///
    /// \brief nearest_search_ Adapter to perform nearest search
    ///
    template<typename ComparisonPolicy>
    std::vector<std::pair<typename ComparisonPolicy::value_type, typename NodeType::data_type>>
    nearest_search_(std::shared_ptr<node_type> node, const data_type& data,
                                           uint_t n, const ComparisonPolicy& calculator)const;

    ///
    /// \brief nearest_search_. Recursion-based adapter to perform nearest serach
    ///
    template<typename ComparisonPolicy, typename PriorityQueueType>
    void do_nearest_search_(std::shared_ptr<node_type> node, const data_type& data,
                                           const ComparisonPolicy& calculator, PriorityQueueType& pq)const;


    template<typename Iterator, typename ComparisonPolicy>
    void create_(Iterator begin, Iterator end, uint_t level, const ComparisonPolicy& comp_policy);


    template<typename Iterator, typename ComparisonPolicy>
    std::shared_ptr<node_type> do_create_(Iterator begin, Iterator end, uint_t level, const ComparisonPolicy& comp_policy);

};

template<typename NodeType>
KDTree<NodeType>::KDTree(uint_t k)
    :
    root_(),
    n_nodes_(0),
    k_(k)
{}

template<typename NodeType>
template<typename Iterator, typename ComparisonPolicy>
KDTree<NodeType>::KDTree(uint_t k, Iterator begin, Iterator end, const ComparisonPolicy& comp_policy)
    :
     KDTree<NodeType>(k)
{
    create_(begin, end, 0, comp_policy);
}

template<typename NodeType>
template<typename ComparisonPolicy>
std::shared_ptr<NodeType>
KDTree<NodeType>::insert(const data_type& data, const ComparisonPolicy& comparison_policy){

    // if the root node is null then
    // simply insert
    if(!root_){
        root_ = std::make_shared<node_type>(0, data, nullptr, nullptr);
        n_nodes_++;
        return root_;
    }

    return insert_(root_, data, comparison_policy, 0);
}

template<typename NodeType>
template<typename ComparisonPolicy>
std::vector<std::pair<typename ComparisonPolicy::value_type, typename NodeType::data_type>>
KDTree<NodeType>::nearest_search_(std::shared_ptr<node_type> node, const data_type& data,
                                     uint_t n, const ComparisonPolicy& calculator)const{

    typedef std::pair<typename ComparisonPolicy::value_type, std::shared_ptr<node_type>> pair_value_type;

    struct comparison
    {
        //typedef std::greater<typename ComparisonPolicy::value_type> compare_op;
        bool operator()(const pair_value_type& v1, const pair_value_type& v2)const{
            return v1.first > v2.first;
        }
    };

    // initialize a min-heap
    //std::priority_queue<value_type, decltype(compare)> pq;
    cubeai::containers::FixedSizeMinPriorityQueue<pair_value_type, comparison> pq(n);

    /// Before starting our search, we need to initialize
    /// the priority queue by adding a “guard”: a tuple
    /// containing infinity as distance, a value that will be
    /// larger than any other distance computed, and so
    /// it will be the first tuple removed from the queue
    /// if we find at least n points in the tree
    pq.push(std::make_pair(std::numeric_limits<typename ComparisonPolicy::value_type>::max(),std::shared_ptr<node_type>()));

    do_nearest_search_(node, data, calculator, pq);

    auto peek = pq.top();

    /// ... if its top element is still at an infinite
    /// distance, we need to remove it, because it
    /// means we have added less than n elements
    /// to the queue

    if(!peek.second){
        pq.pop();
    }

    std::vector<std::pair<typename ComparisonPolicy::value_type, typename NodeType::data_type>> result;
    result.reserve(pq.size());
    while(!pq.empty()){
        auto item = pq.top_and_pop();
        result.push_back({item.first, item.second->data});
    }

    // loop over the priority queue and collect
    // the points we found

    return result;
}

template<typename NodeType>
template<typename ComparisonPolicy, typename PriorityQueueType>
void
KDTree<NodeType>::do_nearest_search_(std::shared_ptr<node_type> node, const data_type& data,
                                     const ComparisonPolicy& calculator, PriorityQueueType& pq)const{


    if(!node){
        // nothing to do with the queue
        return;
    }
    else{

        // compute the distance between the target
        // and the data hold by the node
        auto dist = calculator.evaluate(node->data, data);

        // insrt into the queue
        pq.push(std::make_pair(dist, node));

        if(detail::compare(node, data, k_) < 0){

            auto close_branch = node->left;
            do_nearest_search_(close_branch, data, calculator, pq );

            auto peek = pq.top();

            if(detail::split_distance(node, data, k_) < peek.first){
                do_nearest_search_(node->right, data, calculator, pq );
            }
        }
        else{

            auto close_branch = node->right;
            do_nearest_search_(close_branch, data, calculator, pq );

            auto peek = pq.top();

            if(cubeai::containers::detail::split_distance(node, data, k_) < peek.first){
                do_nearest_search_(node->left, data, calculator, pq );
            }
        }
    }
}

template<typename NodeType>
template<typename DistanceCalculator>
std::shared_ptr<NodeType>
KDTree<NodeType>::search_(std::shared_ptr<node_type> node, const data_type& data, const DistanceCalculator& calculator)const{

    if(!node){
        return nullptr;
    }

    if(calculator(node->data, data)){
        return node;
    }
    else if(detail::compare(node, data, k_) < 0){
        return search_(node->left, data, calculator);
    }
    else{
        return search_(node->right, data, calculator);
    }
}

template<typename NodeType>
template<typename ComparisonPolicy>
std::shared_ptr<NodeType>
KDTree<NodeType>::insert_(std::shared_ptr<node_type>& node, const data_type& data,
                             const ComparisonPolicy& calculator, uint_t level){

    if(!node){
        node = std::make_shared<node_type>(level, data, nullptr, nullptr);
        n_nodes_++;
        return node;
    }


     if(calculator(node->data, data)){

         // we found the data increase the counter
         node->n_copies += 1;
         return node;
     }
    else if(detail::compare(node, data, k_) < 0){
        return insert_(node->left, data, calculator, node->level + 1);
    }


    return insert_(node->right, data, calculator, node->level + 1);

}

template<typename NodeType>
template<typename Iterator, typename ComparisonPolicy>
void
KDTree<NodeType>::create_(Iterator begin, Iterator end, uint_t level, const ComparisonPolicy& comp_policy){


    auto n_points = std::distance(begin, end);

    // nothing to do if no points
    // are given
    if(n_points == 0){
        return ;
    }

    if(n_points == 1){
        auto data = *begin;

        // create the root
        root_ = std::make_shared<NodeType>(level, data, nullptr, nullptr);
        ++n_nodes_;
        return;
    }


    // otherwise partition the range
    auto [median, left, right] = detail::partiion_on_median(std::forward<Iterator>(begin), std::forward<Iterator>(end), level, k_);

    // create root
    root_ = std::make_shared<NodeType>(level, median, nullptr, nullptr);
    ++n_nodes_;

    // create left and right subtrees
    auto left_tree = do_create_(left.first, left.second, level + 1, comp_policy);

    // create left and right subtrees
    auto right_tree = do_create_(right.first, right.second, level + 1, comp_policy);

    root_->left = left_tree;
    root_->right = right_tree;
}

template<typename NodeType>
template<typename Iterator, typename ComparisonPolicy>
std::shared_ptr<NodeType>
KDTree<NodeType>::do_create_(Iterator begin, Iterator end, uint_t level, const ComparisonPolicy& comp_policy){

    auto n_points = std::distance(begin, end);

    // nothing to do if no points
    // are given
    if(n_points == 0){
        return nullptr;
    }

    if(n_points == 1){
        auto data = *begin;

        // check if the data exist
        auto found = search_(root_, data, comp_policy);

        if(found != nullptr){
            found->n_copies += 1;
            return found;
        }

        ++n_nodes_;
        return std::make_shared<NodeType>(level, data, nullptr, nullptr);
    }

    // otherwise partition the range
    auto [median, left, right] = detail::partiion_on_median(std::forward<Iterator>(begin), std::forward<Iterator>(end), level, k_);
            //detail::partiion_on_median(begin, end, level, k_);

    auto left_tree = do_create_(left.first, left.second, level + 1, comp_policy);
    auto right_tree = do_create_(right.first, right.second, level + 1, comp_policy);
    ++n_nodes_;
    return std::make_shared<NodeType>(level, median, left_tree, right_tree);

}


}

}

#endif // KD_TREE_H
