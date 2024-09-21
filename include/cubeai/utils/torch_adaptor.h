#ifndef TORCH_STATE_ADAPTOR_H
#define TORCH_STATE_ADAPTOR_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"

#include <torch/torch.h>
#include <vector>

namespace cubeai{
namespace torch_utils {


///
/// \brief The TorchStateAdaptor struct
///
struct TorchAdaptor{


    typedef torch_tensor_t value_type;

    torch_tensor_t operator()(real_t value)const;
    torch_tensor_t operator()(const std::vector<real_t>& data)const;
    torch_tensor_t operator()(const std::vector<int>& data)const;

    value_type stack(const std::vector<value_type>& values)const;

    template<typename T>
    static std::vector<T> to_vector(torch_tensor_t tensor);
};

}
}

#endif
#endif // TORCH_STATE_ADAPTOR_H
