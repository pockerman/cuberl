#include "cubeai/base/cubeai_config.h"


#ifdef USE_PYTORCH

#include "cubeai/utils/torch_adaptor.h"
#include <torch/torch.h>
#include <vector>

namespace cubeai{
namespace torch_utils{

	

TorchAdaptor::value_type 
TorchAdaptor::to_torch(const std::vector<real_t>& data, DeviceType dtype){
	auto device_ =  dtype != DeviceType::CPU ? torch::kCUDA : torch::kCPU;
	auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device_);
    torch::Tensor d = torch::from_blob(	const_cast<real_t*>(data.data()),
										{static_cast<long int>(data.size())}, options).clone(); //(data);
	return d;
}

TorchAdaptor::value_type 
TorchAdaptor::to_torch(const std::vector<float_t>& data, DeviceType dtype){
	auto device_ =  dtype != DeviceType::CPU ? torch::kCUDA : torch::kCPU;
	auto options = torch::TensorOptions().dtype(torch::kFloat).device(device_);
    torch::Tensor d = torch::from_blob(	const_cast<float_t*>(data.data()),
										{static_cast<long int>(data.size())}, options).clone(); //(data);
	return d;
}

 
TorchAdaptor::value_type 
TorchAdaptor::to_torch(const std::vector<int_t>& data, DeviceType dtype){
	auto device_ =  dtype != DeviceType::CPU ? torch::kCUDA : torch::kCPU;
	auto options = torch::TensorOptions().dtype(torch::kInt).device(device_);
    torch::Tensor d = torch::from_blob(	const_cast<int_t*>(data.data()),
										{static_cast<long int>(data.size())}, options).clone(); //(data);
	return d;
}

 
TorchAdaptor::value_type 
TorchAdaptor::to_torch(const std::vector<lint_t>& data, DeviceType dtype){
	auto device_ =  dtype != DeviceType::CPU ? torch::kCUDA : torch::kCPU;
	auto options = torch::TensorOptions().dtype(torch::kLong).device(device_);
    torch::Tensor d = torch::from_blob(	const_cast<lint_t*>(data.data()),
										{static_cast<long int>(data.size())}, options).clone(); //(data);
	return d;
}

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
TorchAdaptor::stack(const std::vector<value_type>& values, DeviceType dtype)const{

	auto device_ =  dtype != DeviceType::CPU ? torch::kCUDA : torch::kCPU;
    return torch::stack(values, 0).to(device_);
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
std::vector<lint_t>
TorchAdaptor::to_vector(torch_tensor_t tensor){

    auto cont_t = tensor.contiguous();
    return std::vector<lint_t>(cont_t.data_ptr<long int>(),
                              cont_t.data_ptr<long int>() + cont_t.numel());
}

template<>
std::vector<float_t>
TorchAdaptor::to_vector(torch_tensor_t tensor){

    auto cont_t = tensor.contiguous();
    return std::vector<float>(cont_t.data_ptr<float>(),
                              cont_t.data_ptr<float>() + cont_t.numel());
}

template<>
std::vector<real_t>
TorchAdaptor::to_vector(torch_tensor_t tensor){

    auto cont_t = tensor.contiguous();
    return std::vector<real_t>(cont_t.data_ptr<double>(),
                              cont_t.data_ptr<double>() + cont_t.numel());
}

}

}
#endif


