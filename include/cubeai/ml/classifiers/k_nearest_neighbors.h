#ifndef K_NEAREST_NEIGHBORS_H
#define K_NEAREST_NEIGHBORS_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/cubeai_concepts.h"
#include "cubeai/utils/cubeai_traits.h"
#include "cubeai/base/cubeai_config.h"
#include "cubeai/data_structs/kd_tree.h"
#include "cubeai/maths/matrix_utilities.h"


#ifdef CUBEAI_DEBUG
#include <cassert>
#endif

#include <string>
#include <memory>
#include <utility>

namespace cubeai{
namespace ml{
namespace classifiers {

///
///
///
template<typename PointType>
class KNearestNeighbors
{
public:

    ///
    /// \brief get_class_label
    /// \param counters
    /// \return
    ///
    static uint_t get_class_label_from_counters(const std::map<uint_t, uint_t>& counters);

    ///
    /// \brief The Node struct
    ///
    struct Node
    {
        typedef std::pair<PointType, uint_t> data_type;
        typedef data_type value_type;

        uint_t level;
        data_type data;
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;
        uint_t n_copies{0};

        ///
        /// \brief Constructor
        ///
        Node()=default;

        ///
        /// \brief KDTreeNode
        /// \param level
        /// \param left_
        /// \param right_
        ///
        Node(uint_t level, const data_type& data_,
                   std::shared_ptr<Node> left_,  std::shared_ptr<Node> right_);

        ///
        /// \brief get_key. Returns the key that corresponds to the i-th coordinate
        ///
        value_type get_key(uint_t i)const noexcept{return data.first[i];}

        ///
        /// \brief get_point_key
        /// \param level
        /// \param k
        /// \param point
        /// \return
        ///
        typename utils::vector_value_type_trait<PointType>::value_type
        get_point_key(const uint_t level, const uint_t k, const data_type& point ){

            // extract the point value at index level mod k
            // zero-based indexing is assumed
            auto j = level % k;
            return point.first[j];
        }

    };

    ///
    /// \brief KNearestNeighbors
    /// \param n_neighbors
    ///
    KNearestNeighbors(uint_t dim);

    ///
    ///
    ///
    template<typename T, typename ComparisonPolicy>
    void fit(const DynMat<T>& data, const DynVec<uint_t>& labels, const ComparisonPolicy& comp_policy);

    ///
    /// \brief predict. Predict the class of the given point
    /// \param p
    /// \param k
    /// \return
    ///
    template<cubeai::utils::concepts::is_default_constructible SimilarityPolicy>
    uint_t predict(const PointType& p, uint_t k)const;

private:

    ///
    /// \brief tree_
    ///
    cubeai::contaiers::KDTree<Node> tree_;

};

template<typename PointType>
uint_t
KNearestNeighbors<PointType>::get_class_label_from_counters(const std::map<uint_t, uint_t>& counters){

    auto label = cubeai::CubeAIConsts::INVALID_SIZE_TYPE;
    auto counter = 0;

    auto begin = counters.begin();
    auto end = counters.end();

    while( begin != end){

        auto label_idx = begin->first;
        auto counters = begin->second;

        if(counters > counter){
            counter = counters;
            label = label_idx;
        }
    }

    return label;
}

template<typename PointType>
KNearestNeighbors<PointType>::KNearestNeighbors(uint_t dim)
    :
      tree_(dim)
{}

template<typename PointType>
template<typename T,  typename ComparisonPolicy>
void
KNearestNeighbors<PointType>::fit(const DynMat<T>& data, const DynVec<uint_t>& labels, const  ComparisonPolicy& comp_policy){

    for(uint_t r=0; r<data.rows(); ++r){

        PointType p = maths::get_row(data, r);
        uint_t label = labels[r];
#ifdef CUBEAI_DEBUG
    assert(tree_.dim() == p.size() && "Data size not equal to k.");
#endif

        tree_.insert(std::make_pair(p, label), comp_policy);
    }
}

template<typename PointType>
 template<cubeai::utils::concepts::is_default_constructible SimilarityPolicy>
uint_t
KNearestNeighbors<PointType>::predict(const PointType& p, uint_t k)const{

    SimilarityPolicy policy;
    auto result = tree_.nearest_search(p, k, policy);

    std::map<uint_t, uint_t> counters;

    for(auto& item:result){

        auto label = item.second;

        auto label_itr = counters.find(label);

        if(label_itr != counters.end()){
            counters[label] += 1;

        }
        else{
             counters[label] = 1;
        }
    }

    return get_class_label_from_counters(counters);
}

template<typename PointType>
KNearestNeighbors<PointType>::Node::Node(uint_t level_, const data_type& data_,
                                         std::shared_ptr<Node> left_,  std::shared_ptr<Node> right_)
    :
     level(level_),
     data(data_),
     left(left_),
     right(right_),
     n_copies(1)
{}

}
}
}

#endif // K_NEAREST_NEIGHBORS_H
