#ifndef K_NEAREST_NEIGHBORS_H
#define K_NEAREST_NEIGHBORS_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_consts.h"
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
#include <chrono>
#include <ostream>

namespace cubeai{
namespace ml{
namespace classifiers {

struct TrainResult
{
    uint_t n_examples;
    std::chrono::duration<real_t> training_time;

    std::ostream& print(std::ostream& out)const;

};

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
    /// \brief Fit the data
    ///
    template<typename T, typename ComparisonPolicy>
    TrainResult fit(const DynMat<T>& data, const DynVec<uint_t>& labels, const ComparisonPolicy& comp_policy);

    ///
    /// \brief Fit the dataset. Makes a copy of the dataset
    ///
    template<typename TabularDataSetType, typename SimilarityPolicy>
    TrainResult fit(const TabularDataSetType& data, const SimilarityPolicy& comp_policy);

    ///
    /// \brief predict. Predict the class of the given point
    /// \param p
    /// \param k
    /// \return
    ///
    template<cubeai::utils::concepts::is_default_constructible SimilarityPolicy>
    uint_t predict(const PointType& p, uint_t k)const;

    ///
    ///
    ///
    template<cubeai::utils::concepts::is_default_constructible SimilarityPolicy>
    std::vector<std::pair<typename SimilarityPolicy::value_type, typename KNearestNeighbors<PointType>::Node::data_type>>
    nearest_k_points(const PointType& p, uint_t k)const;

private:

    ///
    /// \brief tree_
    ///
    cubeai::containers::KDTree<Node> tree_;

};

template<typename PointType>
uint_t
KNearestNeighbors<PointType>::get_class_label_from_counters(const std::map<uint_t, uint_t>& counters){

    auto label = cubeai::CubeAIConsts::INVALID_SIZE_TYPE;
    uint_t counter = 0;

    auto begin = counters.begin();
    auto end = counters.end();

    while( begin != end){

        auto label_idx = begin->first;
        auto counters = begin->second;

        if(counters > counter){
            counter = counters;
            label = label_idx;
        }

        ++begin;
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
TrainResult
KNearestNeighbors<PointType>::fit(const DynMat<T>& data, const DynVec<uint_t>& labels, const  ComparisonPolicy& comp_policy){

    auto start = std::chrono::steady_clock::now();
    for(uint_t r=0; r<data.rows(); ++r){

        PointType p = maths::get_row(data, r);
        uint_t label = labels[r];
#ifdef CUBEAI_DEBUG
    assert(tree_.dim() == p.size() && "Data size not equal to k.");
#endif

        tree_.insert(std::make_pair(p, label), comp_policy);
    }

    auto end = std::chrono::steady_clock::now();

    TrainResult train_result = {data.rows(), end - start};
    return train_result;
}

template<typename PointType>
template<typename TabularDataSetType, typename SimilarityPolicy>
TrainResult
KNearestNeighbors<PointType>::fit(const TabularDataSetType& data, const SimilarityPolicy& sim_policy){

    auto start = std::chrono::steady_clock::now();

    auto comp_policy = [](const auto& v1, const auto& v2, uint_t idx){
        return v1.first[idx] < v2.first[idx];
    };

    // get a copy of the data
    auto copy_data = data.copy_data();

    tree_.build(copy_data.begin(), copy_data.end(), sim_policy, comp_policy);
    auto end = std::chrono::steady_clock::now();

    TrainResult train_result = {data.n_rows(), end - start};
    return train_result;
}

template<typename PointType>
template<cubeai::utils::concepts::is_default_constructible SimilarityPolicy>
uint_t
KNearestNeighbors<PointType>::predict(const PointType& p, uint_t k)const{

    SimilarityPolicy policy;
    auto data = std::make_pair(p, CubeAIConsts::INVALID_SIZE_TYPE);
    auto result = tree_.nearest_search(data, k, policy);

    std::map<uint_t, uint_t> counters;

    for(auto& item:result){

        auto label = item.second.second;

        auto label_itr = counters.find(label);

        if(label_itr != counters.end()){
            counters[label] += 1;

        }
        else{
             counters.insert({label, 1});
        }
    }

    return get_class_label_from_counters(counters);
}

template<typename PointType>
template<cubeai::utils::concepts::is_default_constructible SimilarityPolicy>
std::vector<std::pair<typename SimilarityPolicy::value_type, typename KNearestNeighbors<PointType>::Node::data_type>>
KNearestNeighbors<PointType>::nearest_k_points(const PointType& p, uint_t k)const{
    SimilarityPolicy policy;
    auto data = std::make_pair(p, CubeAIConsts::INVALID_SIZE_TYPE);
    auto result = tree_.nearest_search(data, k, policy);
    return result;
}

inline
std::ostream&
TrainResult::print(std::ostream& out)const{
    out<<"Trained examples="<<n_examples<<std::endl;
    out<<"Total training time="<<training_time.count()<<"secs"<<std::endl;
    return out;
}

inline
std::ostream& operator<<(std::ostream& out, const TrainResult& result){
    return result.print(out);
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
