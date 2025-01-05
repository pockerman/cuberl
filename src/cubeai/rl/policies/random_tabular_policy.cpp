#include "cubeai/rl/policies/random_tabular_policy.h"
#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_config.h"

#ifdef CUBERL_DEBUG
#include <cassert>
#endif

#include <vector>

namespace cuberl {
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
#ifdef CUBERL_DEBUG
    assert(state_idx < mat.size() && "Invalid state index. Should be state_idx < q_map.size()");
#endif

    return (*this)(mat[state_idx]);

}

template<>
uint_t
RandomTabularPolicy::operator()(const DynMat<real_t>& mat,
                                uint_t state_idx)const{
#ifdef CUBERL_DEBUG
    assert(state_idx < mat.size() && "Invalid state index. Should be state_idx < q_map.size()");
#endif

    return (*this)(mat.row(state_idx));

}

}
}
}
