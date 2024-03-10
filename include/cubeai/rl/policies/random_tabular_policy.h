// SPDX-FileCopyrightText: 2024 <copyright holder> <email>
// SPDX-License-Identifier: Apache-2.0

#ifndef RANDOM_TABULAR_POLICY_H
#define RANDOM_TABULAR_POLICY_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_consts.h"

#include <random>

namespace cubeai {
namespace rl {
namespace policies {

/**
 * @brief class RandomTabularPolicy
 */
class RandomTabularPolicy
{

public:

    /**
     * @brief Constructor
     */
    RandomTabularPolicy()=default;

     /**
     * @brief Constructor Initialize with a seed
     */
    explicit RandomTabularPolicy(uint_t seed);

     /**
     * @brief operator(). Given a
     */
    template<typename MatType>
    uint_t operator()(const MatType& q_map, uint_t state_idx)const;

    /**
     * @brief operator(). Given a vector always returns the position
     * of the maximum occuring element. If the given vector is empty returns
     * CubeAIConsts::invalid_size_type
     */
    template<typename VecTp>
    uint_t operator()(const VecTp& vec)const;

private:

    /**
     * @brief The random engine generator
     */
    mutable std::mt19937 generator_;
};

template<typename VecTp>
uint_t
RandomTabularPolicy::operator()(const VecTp& vec)const{

    std::discrete_distribution<int> distribution(vec.begin(), vec.end());
    return distribution(generator_);

}

}
}
}

#endif // RANDOM_TABULAR_POLICY_H
