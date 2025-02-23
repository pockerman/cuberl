/**
 * Use DQN on Gridworld
 *
 * */
#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"

#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/maths/optimization/optimizer_type.h"
#include "cubeai/maths/optimization/pytorch_optimizer_factory.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"
#include "cubeai/utils/torch_adaptor.h"
#include "cubeai/maths/vector_math.h"

#include "rlenvs/utils/io/csv_file_writer.h"
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

using cuberl::real_t;
using cuberl::uint_t;
using cuberl::float_t;
using cuberl::torch_tensor_t;
using cuberl::rl::RLSerialAgentTrainer;
using cuberl::rl::RLSerialTrainerConfig;
using cuberl::rl::policies::EpsilonGreedyPolicy;
using rlenvscpp::envs::grid_world::Gridworld;

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
									cuberl::rl::policies::EpsilonDecayOption::NONE);

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
				rand_vec = cuberl::maths::randomize(rand_vec, a, b, 64);
				rand_vec = cuberl::maths::divide(rand_vec, 10.0);
				
				// randomize the flattened observation
				obs = cuberl::maths::add(obs, rand_vec);
				auto torch_state = cuberl::utils::pytorch::TorchAdaptor::to_torch(obs, 
				                                                                  cuberl::DeviceType::CPU);
                
                // get the qvals
                auto qvals = qnet(torch_state);
                auto action_idx = policy(qvals, cuberl::torch_tensor_value_type<float_t>());
				
				BOOST_LOG_TRIVIAL(info)<<"Action selected: "<<action_idx<<std::endl;

				// step in the environment
                time_step = env.step(action_idx);
				obs = flattened_observation(time_step.observation());
				// randomize the flattened observation
				obs = cuberl::maths::add(obs, rand_vec);
				
                torch_state = cuberl::utils::pytorch::TorchAdaptor::to_torch(obs, 
				                                                             cuberl::DeviceType::CPU);

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
				
				loss.backward();
				
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
			
			losses.push_back(cuberl::maths::mean(epoch_loss.begin(),
													epoch_loss.end()));
        }

        // save the rewards per episode for visualization
        // purposes
        auto filename = std::string("experiments/") + EXPERIMENT_ID;
        filename += "/dqn_grid_world_policy_rewards.csv";
        rlenvscpp::utils::io::CSVWriter csv_writer(filename, 
		                                           rlenvscpp::utils::io::CSVWriter::default_delimiter());
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

	std::cout<<"This example requires PyTorch"<<std::endl;
	std::cout<<"Reconfigure cuberl with USE_PYTORCH flag turned ON."<<std::endl;
    return 0;
}
#endif
