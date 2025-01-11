#ifndef TORCH_STATE_ADAPTOR_H
#define TORCH_STATE_ADAPTOR_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"

#include <torch/torch.h>
#include <vector>

namespace cuberl{
namespace utils {
namespace pytorch{


///
/// \brief The TorchStateAdaptor struct
///
struct TorchAdaptor{


    typedef torch_tensor_t value_type;
	
	static value_type to_torch(const std::vector<real_t>& data, 
	                           DeviceType dtype=DeviceType::CPU, bool requires_grad=false);
	static value_type to_torch(const std::vector<float_t>& data, 
	                           DeviceType dtype=DeviceType::CPU, bool requires_grad=false);
	static value_type to_torch(const std::vector<int_t>& data, 
	                           DeviceType dtype=DeviceType::CPU, bool requires_grad=false);
	static value_type to_torch(const std::vector<lint_t>& data, 
	                           DeviceType dtype=DeviceType::CPU, bool requires_grad=false);
	static value_type to_torch(const std::vector<bool>& data, 
	                           DeviceType dtype=DeviceType::CPU, bool requires_grad=false);
	
	template<typename T>
	static value_type stack(const std::vector<T>& values, DeviceType type=DeviceType::CPU);

    template<typename T>
    static std::vector<T> to_vector(torch_tensor_t tensor);

    torch_tensor_t operator()(real_t value)const;
    torch_tensor_t operator()(const std::vector<real_t>& data)const;
	torch_tensor_t operator()(const std::vector<float_t>& data)const;
    torch_tensor_t operator()(const std::vector<int>& data)const;
	
    value_type stack(const std::vector<value_type>& values, DeviceType type=DeviceType::CPU)const;



};

}
}
}
#endif
#endif // TORCH_STATE_ADAPTOR_H
