# EXample 12: DQN algorithm on Gridworld

In this example, we will train an agent so that it learns to navigate itself in a grid.
Specifically, we will be using the ```Gridworld``` environmant from the book <a href="https://www.manning.com/books/deep-reinforcement-learning-in-action">Deep Reinforcement Learning in Action</a>.
We have implemented this environment in <a herf="https://github.com/pockerman/rlenvs_from_cpp">rlenvs_from_cpp</a>; check the class <a href="https://github.com/pockerman/rlenvs_from_cpp/blob/master/src/rlenvs/envs/grid_world/grid_world_env.h">Gridworld</a>. 

We will use the DQN algorithm, see <a href="https://www.manning.com/books/deep-reinforcement-learning-in-action">Deep Reinforcement Learning in Action</a> and references therein,
in order to train our agent and we will TensorBoard to monitor the training. We will use a static environment configuration in this example something that makes this problem a lot easier to work on.

We will code the same model as is done in the <a href="https://www.manning.com/books/deep-reinforcement-learning-in-action">Deep Reinforcement Learning in Action</a> book so you may also want to follow the code therein.

## Driver code

```
#include "cubeai/base/cubeai_config.h"

#if defined(USE_PYTORCH) && defined(USE_RLENVS_CPP)

#include "cubeai/base/cubeai_types.h"
#include "cubeai/io/csv_file_writer.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/maths/optimization/optimizer_type.h"
#include "cubeai/maths/optimization/pytorch_optimizer_factory.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"
#include "cubeai/utils/torch_adaptor.h"
#include "cubeai/maths/vector_math.h"
#include "rlenvs/envs/grid_world/grid_world_env.h"

#include <boost/log/trivial.hpp>
#include <torch/torch.h>

#include <unordered_map>
#include <iostream>
#include <string>
#include <any>
#include <filesystem>
#include <map>
#include <vector>

namespace rl_example_12{

const std::string EXPERIMENT_ID = "3";

namespace F = torch::nn::functional;

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::float_t;
using cubeai::torch_tensor_t;

using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using cubeai::rl::policies::EpsilonGreedyPolicy;
using rlenvs_cpp::envs::grid_world::Gridworld;

const uint_t l1 = 64;
const uint_t l2 = 150;
const uint_t l3 = 100;
const uint_t l4 = 4;
const uint_t SEED = 42;
const uint_t TOTAL_EPOCHS = 1000;
const uint_t TOTAL_ITRS_PER_EPOCH = 50;
const real_t GAMMA = 0.9;
const real_t EPSILON = 1.0;
const real_t LEARNING_RATE = 1.0e-3;


// The class that models the Policy network to train
class QNetImpl: public torch::nn::Module
{
public:


    QNetImpl();

    torch_tensor_t forward(torch_tensor_t);

    
private:

   torch::nn::Linear fc1_;
   torch::nn::Linear fc2_;
   torch::nn::Linear fc3_;

};


QNetImpl::QNetImpl()
    :
      fc1_(torch::nn::Linear(l1, l2)),
      fc2_(torch::nn::Linear(l2, l3)),
      fc3_(torch::nn::Linear(l3, l4))
{
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
    register_module("fc3", fc3_);
}


torch_tensor_t
QNetImpl::forward(torch_tensor_t x){

    x = F::relu(fc1_->forward(x));
    x = fc2_->forward(x);
    x = F::relu(fc3_->forward(x));
    return x; 
}

// create the model
TORCH_MODULE(QNet);

// 4x4 grid
typedef Gridworld<4> env_type;
typedef env_type::state_type state_type;


std::vector<float_t>
flattened_observation(const state_type& state){
	
	std::vector<float_t> data;
	data.reserve(64);
	
	for(uint_t i=0; i<state.size(); ++i){
		for(uint_t j=0; j<state[i].size(); ++j){
			for(uint_t k=0; k<state[i][j].size(); ++k){
				data.push_back(state[i][j][k]);
			}
		}
	}
	
	return data;
}

}


int main(){

    using namespace rl_example_12;

    try{

        BOOST_LOG_TRIVIAL(info)<<"Starting agent training...";
        BOOST_LOG_TRIVIAL(info)<<"Numebr of episodes to trina: "<<TOTAL_EPOCHS;

        // let's create a directory where we want to
        //store all the results from running a simulation
        std::filesystem::create_directories("experiments/" + EXPERIMENT_ID);

        // set the seed for PyTorch
        torch::manual_seed(SEED);

        BOOST_LOG_TRIVIAL(info)<<"Creating the environment...";

        // create a 4x4 grid
        auto env = env_type();

        std::unordered_map<std::string, std::any> options;
        env.make("v1", options);

        BOOST_LOG_TRIVIAL(info)<<"Done...";
        BOOST_LOG_TRIVIAL(info)<<"Environment name: "<<env.name;
        BOOST_LOG_TRIVIAL(info)<<"Number of actions available: "<<env.n_actions();
        BOOST_LOG_TRIVIAL(info)<<"Number of states available: "<<env.n_states();


        // the network to train for the q values
        QNet qnet;

        auto optimizer_ptr = std::make_unique<torch::optim::Adam>(qnet->parameters(),
                                                                  torch::optim::AdamOptions(LEARNING_RATE));

        // we will use an epsilon-greedy policy
        EpsilonGreedyPolicy policy(	EPSILON, 
									SEED, 
									cubeai::rl::policies::EpsilonDecayOption::NONE);

        // the loss function to use
        auto loss_fn = torch::nn::MSELoss();

        std::vector<real_t> losses;
        losses.reserve(TOTAL_EPOCHS);

        // loop over the epochs
        for(uint_t epoch=0; epoch < TOTAL_EPOCHS; ++epoch){

            BOOST_LOG_TRIVIAL(info)<<"Starting epoch: "<<epoch<<std::endl;

            // for every new epoch we reset the environment
            auto time_step = env.reset();
            auto done = false;
			uint_t step_counter = 0;
			std::vector<real_t> epoch_loss;
			std::vector<float_t> rand_vec(64, 0.0);
			epoch_loss.reserve(TOTAL_ITRS_PER_EPOCH);
            while(!done){

				auto obs = flattened_observation(time_step.observation());
				
				float_t a = 0.0;
				float_t b = 1.0;
				rand_vec = cubeai::maths::randomize(rand_vec, a, b, 64);
				rand_vec = cubeai::maths::divide(rand_vec, 10.0);
				
				// randomize the flattened observation
				obs = cubeai::maths::add(obs, rand_vec);
				auto torch_state = cubeai::torch_utils::TorchAdaptor::to_torch(obs, cubeai::DeviceType::CPU);
                
                // get the qvals
                auto qvals = qnet(torch_state);
                auto action_idx = policy(qvals, cubeai::torch_tensor_value_type<float_t>());
				
				BOOST_LOG_TRIVIAL(info)<<"Action selected: "<<action_idx<<std::endl;

				// step in the environment
                time_step = env.step(action_idx);
                torch_state = state_to_torch_tensor(time_step.observation());

                // tell the model that we don't use grad here
                qnet->eval();
                auto new_q_vals = qnet(torch_state);
				
				// we are training again
				qnet->train();

                // find the maximum
                auto max_q = torch::max(new_q_vals);
                auto reward = time_step.reward();
				
				BOOST_LOG_TRIVIAL(info)<<"Reward: "<<reward;
				
				// update done
				done = time_step.done();
				
				if(done){ //#Q
					//done = true;
					BOOST_LOG_TRIVIAL(info)<<"Reward: "<<reward;
					BOOST_LOG_TRIVIAL(info)<<"Finishing epoch at step: "<<step_counter;
				}
				
					
                auto y = torch::tensor({reward});
                if(reward == -1.0){
                    y +=  max_q * GAMMA;
                }

                // the target according to Qvals
                auto X = torch::tensor({qvals.squeeze()[action_idx].item<float_t>()});
				
                // calculate the loss
                auto loss = loss_fn(X, y); 
                optimizer_ptr -> zero_grad();
                optimizer_ptr -> step();

                BOOST_LOG_TRIVIAL(info)<<"Loss at epoch: "<<loss.item<real_t>();
                epoch_loss.push_back(loss.item<real_t>());
				step_counter += 1;
            }

            BOOST_LOG_TRIVIAL(info)<<"Epoch finished...";
			
			// get the epsilon
			auto eps = policy.eps_value();
			if(eps > 0.1){
				eps -= 1/static_cast<real_t>(TOTAL_EPOCHS);
				policy.set_eps_value(eps);
			}
			
			losses.push_back(cubeai::maths::mean(epoch_loss.begin(),
													epoch_loss.end()));
        }

        // save the rewards per episode for visualization
        // purposes
        auto filename = std::string("experiments/") + EXPERIMENT_ID;
        filename += "/dqn_grid_world_policy_rewards.csv";
        cubeai::io::CSVWriter csv_writer(filename, cubeai::io::CSVWriter::default_delimiter());
        csv_writer.open();
		csv_writer.write_column_vector(losses);
        
        // save the policy also so that we can load it and check
        // use it
        auto policy_model_filename = std::string("experiments/") + EXPERIMENT_ID;
        policy_model_filename += std::string("/dqn_grid_world_policy.pth");
        torch::serialize::OutputArchive archive;
        qnet -> save(archive);
        archive.save_to(policy_model_filename);

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

	std::cout<<"This example requires PyTorch and rlenvscpplib."<<std::endl;
	std::cout<<"Reconfigure cuberl with USE_PYTORCH and USE_RLENVS_CPP flags turned ON."<<std::endl;
    return 0;
}
#endif

```

Running the code above produces the following output:

```
[2024-09-22 15:21:59.327047] [0x00007f1454f2f000] [info]    Starting agent training...
[2024-09-22 15:21:59.327062] [0x00007f1454f2f000] [info]    Numebr of episodes to trina: 1000
[2024-09-22 15:22:00.527724] [0x00007f1454f2f000] [info]    Creating the environment...
[2024-09-22 15:22:00.528006] [0x00007f1454f2f000] [info]    Done...
[2024-09-22 15:22:00.528012] [0x00007f1454f2f000] [info]    Environment name: Gridworld
[2024-09-22 15:22:00.528018] [0x00007f1454f2f000] [info]    Number of actions available: 4
[2024-09-22 15:22:00.528022] [0x00007f1454f2f000] [info]    Number of states available: 16
[2024-09-22 15:22:00.554437] [0x00007f1454f2f000] [info]    Starting epoch: 0

[2024-09-22 15:22:00.599219] [0x00007f1454f2f000] [info]    Action selected: 3

[2024-09-22 15:22:00.601347] [0x00007f1454f2f000] [info]    Reward: -1
[2024-09-22 15:22:00.607770] [0x00007f1454f2f000] [info]    Loss at epoch: 1.02671
[2024-09-22 15:22:00.608588] [0x00007f1454f2f000] [info]    Action selected: 2

[2024-09-22 15:22:00.608693] [0x00007f1454f2f000] [info]    Reward: -1
[2024-09-22 15:22:00.608754] [0x00007f1454f2f000] [info]    Loss at epoch: 0.989844
[2024-09-22 15:22:00.608844] [0x00007f1454f2f000] [info]    Action selected: 3

[2024-09-22 15:22:00.608925] [0x00007f1454f2f000] [info]    Reward: -1
[2024-09-22 15:22:00.608974] [0x00007f1454f2f000] [info]    Loss at epoch: 1.01939
[2024-09-22 15:22:00.609058] [0x00007f1454f2f000] [info]    Action selected: 3

[2024-09-22 15:22:00.609138] [0x00007f1454f2f000] [info]    Reward: -1
[2024-09-22 15:22:00.609183] [0x00007f1454f2f000] [info]    Loss at epoch: 1.02671
[2024-09-22 15:22:00.609268] [0x00007f1454f2f000] [info]    Action selected: 2
...
```

The average per epoch loss is shown in the figure below

| ![average-per-epoch-loss](./average_loss.png) |
|:--:|
| **Figure: Average loss per epoch.**|

