// SPDX-FileCopyrightText: 2024 <copyright holder> <email>
// SPDX-License-Identifier: Apache-2.0

#ifndef RANDOM_TABULAR_POLICY_H
#define RANDOM_TABULAR_POLICY_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_consts.h"
#include "cubeai/base/cubeai_config.h"
#include "cubeai/utils/torch_adaptor.h"

#ifdef USE_PYTORCH
#include <torch/torch.h>
#endif

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

#ifdef USE_PYTORCH
    uint_t operator()(const torch_tensor_t& vec)const;
#endif

    /**
     * @brief operator(). Given a vector always returns the position
     * of the maximum occuring element. If the given vector is empty returns
     * CubeAIConsts::invalid_size_type
     */
    template<typename VecTp>
    uint_t operator()(const VecTp& vec)const;


    /**
     * @brief any actions the policy should perform
     * on the given episode index
     */
    void on_episode(uint_t)noexcept{}

    /**
     * @brief Reset the policy
     * */
    void reset()noexcept{}

private:

    /**
     * @brief The random engine generator
     */
    mutable std::mt19937 generator_;
};

#ifdef USE_PYTORCH
inline
uint_t
RandomTabularPolicy::operator()(const torch_tensor_t& vec)const{

    auto vector = cubeai::utils::pytorch::TorchAdaptor::to_vector<real_t>(vec);
    std::discrete_distribution<int> distribution(vector.begin(), vector.end());
    return distribution(generator_);

}
#endif

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
