#include "cubeai/base/cubeai_config.h"
#include "cubeai/utils/torch_adaptor.h"

#ifdef USE_PYTORCH

namespace cubeai{
namespace torch_utils{


torch_tensor_t
TorchAdaptor::operator()(real_t value)const{
    return operator()(std::vector<real_t>(1, value));
}

torch_tensor_t
TorchAdaptor::operator()(const std::vector<real_t>& data)const{
    return torch::tensor(data);
}

torch_tensor_t
TorchAdaptor::operator()(const std::vector<int>& data)const{
    return torch::tensor(data);
};

TorchAdaptor::value_type
TorchAdaptor::stack(const std::vector<value_type>& values)const{

    return torch::stack(values, 0);
}


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

template<>
std::vector<float>
TorchAdaptor::to_vector(torch_tensor_t tensor){

    auto cont_t = tensor.contiguous();
    return std::vector<float>(cont_t.data_ptr<float>(),
                              cont_t.data_ptr<float>() + cont_t.numel());
}

template<>
std::vector<double>
TorchAdaptor::to_vector(torch_tensor_t tensor){

    auto cont_t = tensor.contiguous();
    return std::vector<real_t>(cont_t.data_ptr<double>(),
                              cont_t.data_ptr<double>() + cont_t.numel());
}

}

}
#endif


