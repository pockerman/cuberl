/**
  * Example 21: Implements a simple logistic regression
  * model with PyTorch. The example is editted from this
  * https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/basics/logistic_regression/main.cpp
  * GitHub repository. The example uses the MNIST dataset
  *
  *
  *
  *
  */

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_consts.h"

#include <torch/torch.h>
#include <iostream>


namespace example21{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::torch_tensor_t;

//const int64_t input_size = 784;
//const int64_t num_classes = 10;
//const int64_t batch_size = 100;
//const size_t num_epochs = 5;
//const double learning_rate = 0.001;

const uint_t input_size = 784;
const uint_t num_classes = 10;
const uint_t batch_size = 100;
const uint_t num_epochs = 5;
const real_t learning_rate = 0.001;

const std::string MNIST_data_path = cubeai::CubeAIConsts::mnist_data_directory_path(); //"../../../../data/mnist/";


// The regression model
class LogisticRegressionModel: public torch::nn::Module
{

public:

    // constructor. Construct the model by passing
    // the number of features and the number of classes
    LogisticRegressionModel(uint_t input_size, uint_t output_size);

    // forward
    torch_tensor_t forward(torch_tensor_t input);

private:

    torch::nn::Linear linear;
    torch_tensor_t bias_;

};


LogisticRegressionModel::LogisticRegressionModel(uint_t input_size, uint_t output_size)
:
linear(register_module("linear", torch::nn::Linear(input_size, output_size))),
bias_()
{
    bias_ = register_parameter("b", torch::randn(output_size));
}

torch_tensor_t
LogisticRegressionModel::forward(torch_tensor_t input){
    return torch::sigmoid(linear(input) + bias_);
}

}


int main(){

using namespace example21;

 try{

     if(torch::cuda::is_available()){
        std::cout<<"CUDA is available on this machine"<<std::endl;
      }
      else{
        std::cout<<"CUDA is not available on this machine"<<std::endl;
      }

     // Device
     auto cuda_available = torch::cuda::is_available();
     torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);


      // load the data
      // MNIST Dataset (images and labels)
      auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
             .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
             .map(torch::data::transforms::Stack<>());

      // Number of samples in the training set
      auto num_train_samples = train_dataset.size().value();
      std::cout<<cubeai::CubeAIConsts::info_str()<<" Number of train examples"<<num_train_samples<<std::endl;

      // load the test data set
      auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
              .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
              .map(torch::data::transforms::Stack<>());

      // Number of samples in the testset
      auto num_test_samples = test_dataset.size().value();
      std::cout<<cubeai::CubeAIConsts::info_str()<<" Number of test examples"<<num_test_samples<<std::endl;

      // Data loaders
      auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
              std::move(train_dataset), batch_size);

      auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
              std::move(test_dataset), batch_size);

      LogisticRegressionModel model(input_size, num_classes);
      model.to(device);

      // Loss and optimizer
      torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(learning_rate));

      // Set floating point output precision
      std::cout << std::fixed << std::setprecision(4);

      // Train the model
      for (size_t epoch = 0; epoch != num_epochs; ++epoch) {

          // Initialize running metrics
          auto running_loss = 0.0;
          uint_t num_correct = 0;

          for (auto& batch : *train_loader) {

              auto data = batch.data.view({batch_size, -1}).to(device);
              auto target = batch.target.to(device);

              // Forward pass
              auto output = model.forward(data);

              // Calculate loss. Use cross_entropy
              auto loss = torch::nn::functional::cross_entropy(output, target);

              // Update running loss
              running_loss += loss.item<double>() * data.size(0);

              // Calculate prediction
              auto prediction = output.argmax(1);

              // Update number of correctly classified samples
              num_correct += prediction.eq(target).sum().item<int64_t>();

              // Backward pass and optimize
              optimizer.zero_grad();
              loss.backward();
              optimizer.step();
          }

          auto sample_mean_loss = running_loss / num_train_samples;
          auto accuracy = static_cast<real_t>(num_correct) / num_train_samples;

          std::cout<<cubeai::CubeAIConsts::info_str()<< "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
              << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
      }

      std::cout<<cubeai::CubeAIConsts::info_str()<<"Training finished..."<<std::endl;
      std::cout<<cubeai::CubeAIConsts::info_str()<<"Start testing..."<<std::endl;

      // Test the model
      model.eval();
      torch::NoGradGuard no_grad;

      real_t running_loss = 0.0;
      uint_t num_correct = 0;

      for (const auto& batch : *test_loader) {

          auto data = batch.data.view({batch_size, -1}).to(device);
          auto target = batch.target.to(device);

          auto output = model.forward(data);

          auto loss = torch::nn::functional::cross_entropy(output, target);

          running_loss += loss.item<real_t>() * data.size(0);

          auto prediction = output.argmax(1);

          num_correct += prediction.eq(target).sum().item<int64_t>();
      }

      std::cout<<cubeai::CubeAIConsts::info_str()<<"Testing finished..."<<std::endl;;

      auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
      auto test_sample_mean_loss = running_loss / num_test_samples;

      std::cout<<cubeai::CubeAIConsts::info_str()<<"Testset - Loss: "
               << test_sample_mean_loss << ", Accuracy: "
               << test_accuracy <<std::endl;

}
catch(std::exception& e){
   std::cout<<e.what()<<std::endl;
}
catch(...){

   std::cout<<"Unknown exception occured"<<std::endl;
}

return 0;
}
#else
#include <iostream>
int main(){

    std::cout<<"This example requires PyTorch. Reconfigure cubeai with PyTorch support."<<std::endl;
    return 0;
}
#endif
