#include "cubeai/rl/algorithms/utils.h"
#include "cubeai/maths/vector_math.h"
#include <algorithm>

namespace cubeai{
namespace rl{
namespace algos{

std::vector<real_t>
create_discounts_array(real_t base, uint_t npoints){

    std::vector<real_t> points(npoints, 0.0);

    for(uint_t i=0; i<npoints; ++i){
        points[i] = std::pow(base, i);
    }

    return points;
}

real_t
calculate_discounted_return(const std::vector<real_t>& rewards, real_t gamma){

    auto discounted_reward = 0.0;
    auto gammas = maths::logspace(0.0, static_cast<real_t>(rewards.size()),
                                                           rewards.size(), gamma);

    for(uint_t t=0; t<rewards.size(); ++t){
        discounted_reward += std::pow(gammas[t], t)*rewards[t];
    }
    return discounted_reward;
}

#ifdef USE_PYTORCH
std::vector<real_t>
calculate_discounted_returns(torch_tensor_t /*reward*/, torch_tensor_t /*discounts*/, uint_t /*n_workers*/){
    return std::vector<real_t>();
}
#endif

}
}
}
