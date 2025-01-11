#include "cubeai/rl/policies/softmax_policy.h"
#include "cubeai/base/cubeai_config.h"

#include <vector>
#include <algorithm>
#include <iterator>

#ifdef CUBERL_DEBUG
#include <cassert>
#endif

namespace cuberl {
namespace rl {
namespace policies {

template<>
MaxTabularSoftmaxPolicy::output_type
MaxTabularSoftmaxPolicy::operator()(const std::vector<std::vector<real_t>>& q_map,
                             uint_t state_idx)const{

#ifdef CUBERL_DEBUG
    assert(state_idx < q_map.size() && "Invalid state index. Should be state_idx < q_map.size()");
#endif

    return (*this)(q_map[state_idx]);
}


template<>
MaxTabularSoftmaxPolicy::output_type
MaxTabularSoftmaxPolicy::operator()(const DynMat<real_t>& mat,
                                    uint_t state_idx)const{

#ifdef CUBERL_DEBUG
    assert(state_idx < mat.size() && "Invalid state index. Should be state_idx < q_map.size()");
#endif

    return (*this)(mat.row(state_idx));
}

}
}
}

