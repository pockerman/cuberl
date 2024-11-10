#include "cubeai/utils/pytorch_loss_wrapper.h"

#ifdef USE_PYTORCH

#include <torch/torch.h>

#ifdef CUBEAI_DEBUG
#include <cassert>
#endif

namespace cubeai{
namespace torch_utils {

using namespace cubeai::utils;

PyTorchLossWrapper::PyTorchLossWrapper(LossType type)
    :
      type_(type)
{}


torch_tensor_t
PyTorchLossWrapper::calculate(torch_tensor_t output, torch_tensor_t target)const{


    switch(type_)
    {
       case LossType::CROSS_ENTROPY:
       {
           return torch::nn::functional::cross_entropy(output, target);
       }
       case LossType::MSE:
       {
         torch::nn::MSELoss mse;
         return mse(output, target);

       }
#ifdef CUBEAI_DEBUG
    default:
    {
        throw std::logic_error("Invalid PyTorch loss type");
    }
#endif
    }

    return torch_tensor_t();
}

}
}

#endif
