#include "cubeai/rl/algorithms/utils.h"
#include "cubeai/utils/array_utils.h"
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

std::vector<real_t>
calculate_discounted_returns(const std::vector<real_t>& rewards,
                             const std::vector<real_t>& discounts, uint_t n_workers){

    // T
    auto total_time = rewards.size();

    // Return numbers spaced evenly on a log scale.
    // In linear space, the sequence starts at base ** start
    // (base to the power of start) and ends with base ** stop (see endpoint below).
    // The return is the sum of discounted rewards from step until the
    // final step T

    for(uint_t w = 0; w < n_workers; ++w ){

        for(uint_t t=0; t < total_time; ++t){

            // get the subarrays
            auto worker_rewards = rewards[t];

            auto time_discounts = extract_subvector(discounts, total_time - t);
        }

    }

    //returns = np.array([[np.sum(discounts[: total_time - t] * rewards[t:, w]) for t in range(total_time)] for w in range(n_workers)])
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
