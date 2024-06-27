#ifndef PYTORCH_OPTIMIZER_FACTORY_H
#define PYTORCH_OPTIMIZER_FACTORY_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH


#include "cubeai/maths/optimization/optimizer_type.h"
#include <torch/torch.h>
#include <memory>
#include <map>
#include <any>
#include <string>

#ifdef CUBEAI_DEBUG
#include <cassert>
#endif

namespace cubeai {
namespace maths{
namespace optim {
namespace pytorch {


///
///
///
std::unique_ptr<torch::optim::OptimizerOptions>
build_pytorch_optimizer_options(OptimzerType type, const std::map<std::string, std::any>& options);

///
/// \brief build_pytorch_optimizer
/// \param type
/// \param model
/// \return
std::unique_ptr<torch::optim::Optimizer>
build_pytorch_optimizer(OptimzerType type, torch::nn::Module& model, const torch::optim::OptimizerOptions& options);

///
/// \brief build_pytorch_optimizer
/// \param type
/// \param model
/// \return
std::unique_ptr<torch::optim::Optimizer>
build_pytorch_optimizer(OptimzerType type, torch::nn::Module& model, std::unique_ptr<torch::optim::OptimizerOptions>& options){
    return build_pytorch_optimizer(type, model, *options.get());
}

}
}
}
}
#endif
#endif // PYTORCH_OPTIMIZER_FACTORY_H
