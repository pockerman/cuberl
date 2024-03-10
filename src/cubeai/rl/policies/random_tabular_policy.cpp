// SPDX-FileCopyrightText: 2024 <copyright holder> <email>
// SPDX-License-Identifier: Apache-2.0

#include "cubeai/rl/policies/random_tabular_policy.h"
#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_config.h"

#ifdef CUBEAI_DEBUG
#include <cassert>
#endif

namespace cubeai {
namespace rl {
namespace policies {

RandomTabularPolicy::RandomTabularPolicy(uint_t seed)
:
generator_(seed)
{}



template<>
uint_t
RandomTabularPolicy::operator()(const std::vector<std::vector<real_t>>& mat,
                                uint_t state_idx)const{
#ifdef CUBEAI_DEBUG
    assert(state_idx < mat.size() && "Invalid state index. Should be state_idx < q_map.size()");
#endif

    return (*this)(mat[state_idx]);

}

template<>
uint_t
RandomTabularPolicy::operator()(const DynMat<real_t>& mat,
                                uint_t state_idx)const{
#ifdef CUBEAI_DEBUG
    assert(state_idx < mat.size() && "Invalid state index. Should be state_idx < q_map.size()");
#endif

    return (*this)(mat.row(state_idx));

}

}
}
}
