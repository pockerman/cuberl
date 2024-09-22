// SPDX-FileCopyrightText: 2024 <copyright holder> <email>
// SPDX-License-Identifier: Apache-2.0

#ifndef MAX_TABULAR_POLICY_H
#define MAX_TABULAR_POLICY_H

#include "cubeai/base/cubeai_config.h"
#include "cubeai/base/cubeai_types.h"

#ifdef USE_PYTORCH
#include <torch/torch.h>
#endif

namespace cubeai {
namespace rl {
namespace policies {

/**
 * @brief class MaxTabularPolicy
 */
class MaxTabularPolicy
{
public:

    /**
     * @brief The output type of operator()
     */
    typedef uint_t output_type;

    /**
    *  @brief Constructor
    */
    MaxTabularPolicy()=default;

    /**
     * @brief operator(). Given a
     */
    template<typename MatType>
    output_type operator()(const MatType& q_map, uint_t state_idx)const;

#ifdef USE_PYTORCH
     /**
     * @brief operator(). Given a vector always returns the position
     * of the maximum occuring element. If the given vector is empty returns
     * CubeAIConsts::invalid_size_type
     */
    uint_t operator()(const torch_tensor_t& vec)const;
#endif

    /**
     * @brief operator(). Given a vector always returns the position
     * of the maximum occuring element. If the given vector is empty returns
     * CubeAIConsts::invalid_size_type
     */
    template<typename VecTp>
    output_type operator()(const VecTp& q_map)const;

    /**
     * @brief any actions the policy should perform
     * on the given episode index
     */
    void on_episode(uint_t)noexcept{}

    /**
     * @brief Reset the policy
     * */
    void reset()noexcept{}


};


#ifdef USE_PYTORCH
inline
uint_t
MaxTabularPolicy::operator()(const torch_tensor_t& vec)const{
    return torch::argmax(vec).item<uint_t>();
}
#endif


template<typename VecTp>
uint_t
MaxTabularPolicy::operator()(const VecTp& vec)const{

    return std::distance(vec.begin(),
                         std::max_element(vec.begin(),
                                          vec.end()));

}


}
}
}

#endif // MAX_TABULAR_POLICY_H
