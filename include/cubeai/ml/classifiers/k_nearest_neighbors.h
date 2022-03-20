#ifndef K_NEAREST_NEIGHBORS_H
#define K_NEAREST_NEIGHBORS_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/data_structs/kd_tree.h"

#include <string>
#include <memory>

namespace cubeai{
namespace ml{
namespace classifiers {


template<typename PointType, typename ComparisonPolicy>
class KNearestNeighbors
{
public:

    ///
    /// \brief KNearestNeighbors
    /// \param n_neighbors
    ///
    KNearestNeighbors();

    ///
    ///
    ///
    template<typename DataSetType, typename LabelsType>
    void fit(const DataSetType& data, const LabelsType& labels);

    ///
    /// \brief load_and_fit
    /// \param path
    ///
    void load_and_fit(const std::string& path);

    ///
    /// \brief predict. Predict the class of the given point
    /// \param p
    /// \param k
    /// \return
    ///
    uint_t predict(const PointType& p, uint_t k)const;

private:

    struct Node
    {
        typedef DataType data_type;
        typedef typename DataType::value_type value_type;

        data_type data;
        uint_t label;
        uint_t idx;
        std::shared_ptr<Node<data_type>> left;
        std::shared_ptr<Node<data_type>> right;
        uint_t level;
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
                   std::shared_ptr<KDTreeNode<data_type>> left_,  std::shared_ptr<KDTreeNode<data_type>> right_);

        ///
        /// \brief get_key. Returns the key that corresponds to the i-th coordinate
        ///
        value_type get_key(uint_t i)const noexcept{return data[i];}

    };

    ///
    /// \brief tree_
    ///
    cubeai::contaiers::KDTree<Node> tree_;


};

template<typename PointType, typename ComparisonPolicy>
template<typename DataSetType, typename LabelsType>
void
KNearestNeighbors<PointType, ComparisonPolicy>::fit(const DataSetType& data, const LabelsType& labels){

    tree_.create(data);

}

}
}
}

#endif // K_NEAREST_NEIGHBORS_H
