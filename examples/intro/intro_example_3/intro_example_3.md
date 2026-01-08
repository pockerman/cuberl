# Example 3: Using PyTorch C++ API Part 1

Example intro_example_1 demonstrated some functionality
cuberl is heavily based on the C++ API exposed by PyTorch. 
Therefore, this tutorial is mean to show you some basics on how to create
and train PyTorch models using the C++ API. There are many resources you can
consult for further info:

- <a href="https://pytorch.org/cppdocs/">PyTorch C++ API</a>
- <a href="https://pytorch.org/tutorials/advanced/cpp_frontend.html">Using the PyTorch C++ Frontend</a>
- <a href="https://github.com/prabhuomkar/pytorch-cpp/tree/master">pytorch-cpp</a>

In this tutorial we will reproduce the code in <a href="https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/basics/linear_regression/main.cpp">linear regression with PyTorch</a>.
We will also show how to read a json file with cuberl, how to use the ```IterativeAlgorithmController``` class. Finally, we will see how to use the logging utilities from the boost C++ library.
Note that this tutorial requires cuberl to be configured with PyTorch support.


## The driver code

The driver code for this tutorial is shown below. 


@code
#include "cubeai/base/cubeai_config.h"

#if defined(USE_PYTORCH) && defined(USE_RLENVS_CPP)

#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/iteration_counter.h"
#include "cubeai/io/json_file_reader.h"

#include <torch/torch.h>
#include <boost/log/trivial.hpp>

#include <iostream>
#include <random>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <any>
#include <fstream>

namespace intro_example_2
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::io::JSONFileReader;
using cubeai::utils::IterationCounter;

namespace fs = std::filesystem;
const std::string CONFIG = "config.json";

}

int main() {

    using namespace intro_example_2;

    try{

      // load the json configuration
      JSONFileReader json_reader(CONFIG);
      json_reader.open();

      auto experiment_dict = json_reader.template get_value<std::string>("experiment_dict");
      auto experiment_id = json_reader.template get_value<std::string>("experiment_id");

      BOOST_LOG_TRIVIAL(info)<<"Experiment directory: "<<experiment_dict;
      BOOST_LOG_TRIVIAL(info)<<"Experiment id: "<<experiment_id<<std::endl;

      const fs::path EXPERIMENT_DIR_PATH = experiment_dict + experiment_id;

      // the first thing we want to do when monitoring experiments
      // is to create a directory where all data will reside
      std::filesystem::create_directories(experiment_dict + experiment_id);

      const auto input_size = json_reader.template get_value<uint_t>("input_size");
      const auto output_size = json_reader.template get_value<uint_t>("output_size");
      const auto num_epochs = json_reader.template get_value<uint_t>("num_epochs");
      const auto learning_rate = json_reader.template get_value<real_t>("lr");

      // log the hyperparameters
      BOOST_LOG_TRIVIAL(info)<<"Input size: "<<input_size;
      BOOST_LOG_TRIVIAL(info)<<"Output size: "<<output_size;
      BOOST_LOG_TRIVIAL(info)<<"Max epochs: "<<num_epochs;
      BOOST_LOG_TRIVIAL(info)<<"Learning rate: "<<learning_rate;


      // figure out the device we are using
      auto cuda_available = torch::cuda::is_available();
      torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
      if(cuda_available){
        BOOST_LOG_TRIVIAL(info)<<"CUDA available. Training on GPU "<<std::endl;
      }
      else{
        BOOST_LOG_TRIVIAL(info)<<"CUDA is not available. Training on CPU "<<std::endl;
      }

      torch::manual_seed(42);
      // Sample dataset
      auto x_train = torch::randint(0, 10, {15, 1},
                                    torch::TensorOptions(torch::kFloat).device(device));

      auto y_train = torch::randint(0, 10, {15, 1},
                                    torch::TensorOptions(torch::kFloat).device(device));

      // Linear regression model
      torch::nn::Linear model(input_size, output_size);
      model->to(device);

      // Optimizer
      torch::optim::SGD optimizer(model->parameters(),
                                  torch::optim::SGDOptions(learning_rate));

      // Set floating point output precision
      BOOST_LOG_TRIVIAL(info)<<"Start training...";


      IterationCounter iteration_ctrl(num_epochs);
      while(iteration_ctrl.continue_iterations()){

        // Forward pass
          auto output = model->forward(x_train);
          auto loss = torch::nn::functional::mse_loss(output, y_train);

          // Backward pass and optimize
          optimizer.zero_grad();
          loss.backward();
          optimizer.step();

          auto current_iteration = iteration_ctrl.current_iteration_index();
          if ((current_iteration + 1) % 5 == 0) {
            BOOST_LOG_TRIVIAL(info)<< "Epoch [" << (current_iteration + 1) << "/" << num_epochs <<"], Loss: " << loss.item<double>(); //<< "\n";
          }
      }

      BOOST_LOG_TRIVIAL(info)<<"Training is finished... ";

      // let's also save the model
      auto model_filename = std::string(EXPERIMENT_DIR_PATH) + std::string("/linear_regression_model.pth");
      torch::save(model, model_filename);

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

    std::cout<<"This example requires PyTorch and gymfcpp. Reconfigure cuberl with USE_PYTORCH and USE_RLENVS_CPP flags turned ON."<<std::endl;
    return 0;
}
#endif
@endcode

Running the code above produces the following output

```
[2024-05-27 10:15:48.581146] [0x00007f9aaa83d000] [info]    Experiment directory: intro_example_2_experiments/
[2024-05-27 10:15:48.581173] [0x00007f9aaa83d000] [info]    Experiment id: 1

[2024-05-27 10:15:48.581214] [0x00007f9aaa83d000] [info]    Input size: 1
[2024-05-27 10:15:48.581217] [0x00007f9aaa83d000] [info]    Output size: 1
[2024-05-27 10:15:48.581221] [0x00007f9aaa83d000] [info]    Max epochs: 60
[2024-05-27 10:15:48.581238] [0x00007f9aaa83d000] [info]    Learning rate: 0.001
[2024-05-27 10:15:49.777364] [0x00007f9aaa83d000] [info]    CUDA available. Training on GPU 

[2024-05-27 10:15:49.841728] [0x00007f9aaa83d000] [info]    Start training...
[2024-05-27 10:15:49.930313] [0x00007f9aaa83d000] [info]    Epoch [5/60], Loss: 7.19124
[2024-05-27 10:15:49.931668] [0x00007f9aaa83d000] [info]    Epoch [10/60], Loss: 6.83168
[2024-05-27 10:15:49.932512] [0x00007f9aaa83d000] [info]    Epoch [15/60], Loss: 6.65128
[2024-05-27 10:15:49.933157] [0x00007f9aaa83d000] [info]    Epoch [20/60], Loss: 6.56069
[2024-05-27 10:15:49.933783] [0x00007f9aaa83d000] [info]    Epoch [25/60], Loss: 6.51511
[2024-05-27 10:15:49.934423] [0x00007f9aaa83d000] [info]    Epoch [30/60], Loss: 6.49209
[2024-05-27 10:15:49.935069] [0x00007f9aaa83d000] [info]    Epoch [35/60], Loss: 6.48037
[2024-05-27 10:15:49.935710] [0x00007f9aaa83d000] [info]    Epoch [40/60], Loss: 6.47432
[2024-05-27 10:15:49.936349] [0x00007f9aaa83d000] [info]    Epoch [45/60], Loss: 6.47112
[2024-05-27 10:15:49.937007] [0x00007f9aaa83d000] [info]    Epoch [50/60], Loss: 6.46934
[2024-05-27 10:15:49.937671] [0x00007f9aaa83d000] [info]    Epoch [55/60], Loss: 6.46827
[2024-05-27 10:15:49.938189] [0x00007f9aaa83d000] [info]    Epoch [60/60], Loss: 6.46756
[2024-05-27 10:15:49.938197] [0x00007f9aaa83d000] [info]    Training is finished... 
```

