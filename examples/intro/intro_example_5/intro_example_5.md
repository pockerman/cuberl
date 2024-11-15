# Example 5:  Using ```TensorboardServer``` class

Example 1 showed you how to use ```BOOST_LOG_TRIVIAL``` to log output when running a program.
This example utilises the ```TensorboardServer``` class to log values of interest when running
an experiment. We can monitor the experimet  using <a href="https://www.tensorflow.org/tensorboard">tensorboard</a>.

The   ```TensorboardServer``` class is a simple wrapper that exposes three functions

- ```add_scalar```
- ```add_scalars```
- ```add_text```

We will use ```add_scalar``` and ```add_text```. In order to run this example, fire up the server using the ```torchboard_server/start_uvicorn.sh```.
The server listens at port 8002. You can change this however you want just make sure that the port is not used and also update the
variable ```TORCH_SERVER_HOST``` in the code below accordingly. Note that the implementation uses
<a href="https://pytorch.org/docs/stable/_modules/torch/utils/tensorboard/writer.html#SummaryWriter">SummaryWriter</a> class.
Thus you will need to have PyTorch installed on the machine that you run the server.


## The driver code

The driver code for this tutorial is shown below.

```cpp
#include "cubeai/base/cubeai_config.h"

#if defined(USE_PYTORCH)

#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/iteration_counter.h"
#include "cubeai/io/json_file_reader.h"
#include "cubeai/io/tensorboard_server.h"

#include <torch/torch.h>
#include <boost/log/trivial.hpp>

#include <random>
#include <filesystem>
#include <string>

namespace intro_example_4
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::io::JSONFileReader;
using cubeai::io::TensorboardServer;
using cubeai::utils::IterationCounter;

namespace fs = std::filesystem;
const std::string CONFIG = "config.json";
const std::string TORCH_SERVER_HOST = "http://0.0.0.0:8002"

}

int main() {

    using namespace intro_example_4;

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
      auto log_path = json_reader.template get_value<std::string>("log_path");

      // log the hyperparameters
      BOOST_LOG_TRIVIAL(info)<<"Input size: "<<input_size;
      BOOST_LOG_TRIVIAL(info)<<"Output size: "<<output_size;
      BOOST_LOG_TRIVIAL(info)<<"Max epochs: "<<num_epochs;
      BOOST_LOG_TRIVIAL(info)<<"Learning rate: "<<learning_rate;



      BOOST_LOG_TRIVIAL(info)<<"Logging results at "<<logger.get_log_dir_path()<<std::endl;


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

      TensorboardServer logger(TORCH_SERVER_HOST);
      logger.init(log_path);

      logger.add_scalar("lr", learning_rate);
      logger.add_scalar("seed", 42);
      logger.add_scalar("num_epochs", num_epochs);
      logger.add_text("optimizer", "torch::optim::SGD");
      if(cuda_available){
        logger.add_text("device", "GPU");
      }
      else{
        logger.add_text("device", "CPU");
      }

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
            BOOST_LOG_TRIVIAL(info)<< "Epoch [" << (current_iteration + 1) << "/" << num_epochs <<"], Loss: " << loss.item<real_t>();
            logger.add_scalar("Loss/Training", loss.item<real_t>());
          }
      }

      BOOST_LOG_TRIVIAL(info)<<"Training is finished... ";

      // let's also save the model
      auto model_filename = std::string(EXPERIMENT_DIR_PATH) + std::string("/linear_regression_model.pth");
      torch::save(model, model_filename);

      logger.close();

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

    std::cout<<"This example requires PyTorch and gymfcpp. Reconfigure cuberl with USE_PYTORCH flags turned ON."<<std::endl;
    return 0;
}
#endif

```

