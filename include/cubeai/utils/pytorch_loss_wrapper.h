#ifndef PYTORCH_LOSS_WRAPPER_H
#define PYTORCH_LOSS_WRAPPER_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/loss_type.h"
#include <torch/torch.h>




namespace cubeai {
namespace utils{
namespace pytorch {
	
	using namespace cubeai::utils;

///
/// \brief The PyTorchLossWrapper class
///
class PyTorchLossWrapper
{
public:

    ///
    /// \brief PyTorchLossWrapper
    /// \param type
    ///
    PyTorchLossWrapper(LossType type);

    ///
    /// \brief calculate
    /// \param input
    /// \param target
    /// \return
    ///
    torch_tensor_t calculate(torch_tensor_t input, torch_tensor_t target)const;

private:

    ///
    /// \brief type_
    ///
    LossType type_;

};

}
}
}

#endif
#endif // PYTORCH_LOSS_WRAPPER_H
