// SPDX-FileCopyrightText: 2024 <copyright holder> <email>
// SPDX-License-Identifier: Apache-2.0

#include "cubeai/rl/policies/max_tabular_policy.h"
#include "cubeai/base/cubeai_config.h"

#include <vector>
#include <algorithm>
#include <iterator>

#ifdef CUBEAI_DEBUG
#include <cassert>
#endif

namespace cubeai {
namespace rl {
namespace policies {





template<>
uint_t
MaxTabularPolicy::operator()(const std::vector<std::vector<real_t>>& q_map,
                             uint_t state_idx)const{

#ifdef CUBEAI_DEBUG
    assert(state_idx < q_map.size() && "Invalid state index. Should be state_idx < q_map.size()");
#endif

    return (*this)(q_map[state_idx]);
}


template<>
uint_t
MaxTabularPolicy::operator()(const DynMat<real_t>& mat,
                             uint_t state_idx)const{

#ifdef CUBEAI_DEBUG
    assert(state_idx < mat.size() && "Invalid state index. Should be state_idx < q_map.size()");
#endif

    return (*this)(mat.row(state_idx));
}





}
}
}
