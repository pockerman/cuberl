#include "cubeai/rl/algorithms/utils.h"

namespace cubeai{
namespace rl{
namespace algos{


#ifdef USE_PYTORCH
std::vector<real_t>
calculate_discounted_returns(torch_tensor_t reward, torch_tensor_t discounts, uint_t n_workers){
    return std::vector<real_t>();
}
#endif

}
}
}
