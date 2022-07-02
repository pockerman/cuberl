#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "torch/torch.h"

namespace cubeai {
namespace pytorch_uitls{

///
/// \brief mse Compute the mean square error MSE between the two vectors
/// \param vec1
/// \param vec2
/// \return  A torch tensor representing the MSE loss
///
torch_tensor_t
mse(torch_tensor_t vec1, torch_tensor_t vec2);
}
}

#endif
#endif // LOSS_FUNCTIONS_H
