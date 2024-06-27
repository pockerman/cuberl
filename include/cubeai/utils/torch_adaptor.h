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
    typedef torch_tensor_t state_type;

    torch_tensor_t operator()(real_t value)const;
    torch_tensor_t operator()(const std::vector<real_t>& data)const;
    torch_tensor_t operator()(const std::vector<int>& data)const;

    value_type stack(const std::vector<value_type>& values)const;

    template<typename T>
    static std::vector<T> to_vector(torch_tensor_t tensor);
};


template<>
std::vector<int>
TorchAdaptor::to_vector(torch_tensor_t tensor){

    auto cont_t = tensor.contiguous();
    return std::vector<int>(cont_t.data_ptr<int>(),
                              cont_t.data_ptr<int>() + cont_t.numel());
}

template<>
std::vector<uint_t>
TorchAdaptor::to_vector(torch_tensor_t tensor){

    auto cont_t = tensor.contiguous();
    return std::vector<uint_t>(cont_t.data_ptr<long int>(),
                              cont_t.data_ptr<long int>() + cont_t.numel());
}

#ifdef CUBEAI_REAL_TYPE_FLOAT
template<>
std::vector<float>
TorchAdaptor::to_vector(torch_tensor_t tensor){

    auto cont_t = tensor.contiguous();
    return std::vector<float>(cont_t.data_ptr<float>(),
                              cont_t.data_ptr<float>() + cont_t.numel());
}

#else
template<>
std::vector<float>
TorchAdaptor::to_vector(torch_tensor_t tensor){

    auto cont_t = tensor.contiguous();
    return std::vector<float>(cont_t.data_ptr<float>(),
                              cont_t.data_ptr<float>() + cont_t.numel());
}

template<>
std::vector<real_t>
TorchAdaptor::to_vector(torch_tensor_t tensor){

    auto cont_t = tensor.contiguous();
    return std::vector<real_t>(cont_t.data_ptr<real_t>(),
                              cont_t.data_ptr<real_t>() + cont_t.numel());
}
#endif

}
}

#endif
#endif // TORCH_STATE_ADAPTOR_H
