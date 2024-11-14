#include "cubeai/pytorch_utils/loss_functions.h"

#ifdef USE_PYTORCH

namespace cubeai {
namespace pytorch_uitls{

torch_tensor_t
mse(torch_tensor_t vec1, torch_tensor_t vec2){

    auto value_error = vec1 - vec2;
    auto loss = value_error.pow(2).mul(0.5).mean();
    return loss;
}
}
}

#endif

