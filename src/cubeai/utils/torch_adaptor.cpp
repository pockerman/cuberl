#include "cubeai/base/cubeai_config.h"
#include "cubeai/utils/torch_adaptor.h"

#ifdef USE_PYTORCH

namespace cubeai{
namespace torch_utils{


torch_tensor_t
TorchAdaptor::operator()(real_t value)const{
    return this->operator()(std::vector<real_t>(1, value));
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

}

}
#endif


