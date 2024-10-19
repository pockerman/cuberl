/**
 * Use DQN on Gridworld
 *
 * */
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
#include "cubeai/data_structs/experience_buffer.h"
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
#include <tuple>

namespace rl_example_12{

	namespace F = torch::nn::functional;
	
using cubeai::real_t;
using cubeai::uint_t;
using cubeai::float_t;
using cubeai::torch_tensor_t;

using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using cubeai::rl::policies::EpsilonGreedyPolicy;
using cubeai::containers::ExperienceBuffer;
using rlenvs_cpp::envs::grid_world::Gridworld;

	
const std::string EXPERIMENT_ID = "1";

const uint_t l1 = 64;
const uint_t l2 = 150;
const uint_t l3 = 100;
const uint_t l4 = 4;
const uint_t SEED = 42;
const uint_t BATCH_SIZE = 200;
const uint_t EXPERIENCE_BUFFER_SIZE = 1000;
const uint_t TOTAL_EPOCHS = 1000;
const uint_t TOTAL_ITRS_PER_EPOCH = 50;
const real_t GAMMA = 0.9;
const real_t EPSILON = 0.3;
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
typedef env_type::time_step_type time_step_type;

typedef std::tuple<std::vector<float_t>, uint_t, real_t, std::vector<float_t>, bool> experience_tuple_type;

typedef ExperienceBuffer<experience_tuple_type> experience_buffer_type;
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

template<typename T, uint_t index>
std::vector<T> 
get(const std::vector<experience_tuple_type>& experience){
	
	std::vector<T> result;
	result.reserve(experience.size());
	
	auto b = experience.begin();
	auto e = experience.end();
	
	for(; b != e; ++b){
		auto item = *b;
		result.push_back(std::get<index>(item));
	}
	
	return result;
	
}

}


int main(){

    using namespace rl_example_12;
	using namespace cubeai;

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

        
		// initialize the environment using random mode
		std::unordered_map<std::string, std::any> options;
        options["mode"] = std::any(rlenvs_cpp::envs::grid_world::to_string(rlenvs_cpp::envs::grid_world::GridWorldInitType::RANDOM));

        env.make("v0", options);

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
									rl::policies::EpsilonDecayOption::NONE);

        // the loss function to use
        auto loss_fn = torch::nn::MSELoss();

		// track the average loss per epoch
        std::vector<real_t> losses;
        losses.reserve(TOTAL_EPOCHS);
		
		experience_buffer_type experience_buffer(EXPERIENCE_BUFFER_SIZE);


		// hold random values
		std::vector<float_t> rand_vec(64, 0.0);
		float_t a = 0.0;
		float_t b = 1.0;
        
		// loop over the epochs
        for(uint_t epoch=0; epoch < TOTAL_EPOCHS; ++epoch){

            BOOST_LOG_TRIVIAL(info)<<"Starting epoch: "<<epoch<<std::endl;

            // for every new epoch we reset the environment
            auto time_step = env.reset();
            
			uint_t step_counter = 0;
			
			
			// the loss associated with the epoch
			std::vector<real_t> epoch_loss;
			epoch_loss.reserve(TOTAL_ITRS_PER_EPOCH);
            
			auto done = false;
			while(!done){

				auto obs1 = flattened_observation(time_step.observation());
				
				rand_vec = cubeai::maths::randomize(rand_vec, a, b, 64);
				rand_vec = cubeai::maths::divide(rand_vec, 100.0);
				
				// randomize the flattened observation by adding
				// some noise
				obs1 = maths::add(obs1, rand_vec);
				auto torch_state_1 = torch_utils::TorchAdaptor::to_torch(obs1, 
																		 DeviceType::CPU);
                
                // get the qvals
                auto qvals = qnet(torch_state_1);
                auto action_idx = policy(qvals, 
				                         cubeai::torch_tensor_value_type<float_t>());
				
				BOOST_LOG_TRIVIAL(info)<<"Action selected: "<<action_idx<<std::endl;

				// step in the environment
                time_step = env.step(action_idx);
				auto reward = time_step.reward();
				auto step_finished = time_step.done();
				
				auto obs2 = flattened_observation(time_step.observation());
				rand_vec = cubeai::maths::randomize(rand_vec, a, b, 64);
				rand_vec = cubeai::maths::divide(rand_vec, 100.0);
				obs2 = maths::add(obs2, rand_vec);
				
				auto torch_state_2 = torch_utils::TorchAdaptor::to_torch(obs2, 
																		 DeviceType::CPU);
																		 
																		 
				
				experience_tuple_type exp = {obs1, action_idx, reward, obs2, step_finished}; 
				
				// put the observation into the buffer
				experience_buffer.append(exp);
				
				// if we reach the batch size we will do
				// a backward propagation
				if(experience_buffer.size() >= BATCH_SIZE){
					
					std::vector<experience_tuple_type> batch_sample;
					batch_sample.reserve(BATCH_SIZE);
					
					// sample from the experience 
					experience_buffer.sample(BATCH_SIZE, batch_sample, SEED);
					
					// stack the experiences
					auto state_1_batch = get<std::vector<float_t>, 0>(batch_sample);
					auto action_batch  = get<int_t, 1>(batch_sample);
					auto state_2_batch = get<std::vector<float_t>, 3>(batch_sample);
					auto reward_batch  = get<real_t, 2>(batch_sample);
					auto done_batch    = get<bool, 4>(batch_sample);
					
					auto state_1_batch_t = torch_utils::TorchAdaptor::stack(state_1_batch, 
																			cubeai::DeviceType::CPU);
					auto q1 = qnet(state_1_batch_t);
					
					// tell the model that we don't use grad here
					qnet->eval();
					
					auto state_2_batch_t = torch_utils::TorchAdaptor::stack(state_2_batch, 
																			cubeai::DeviceType::CPU);
					auto q2 = qnet(state_2_batch_t);
				
					// we are training again
					qnet->train();
					
					auto reward_batch_t = torch_utils::TorchAdaptor::to_torch(reward_batch, 
																			  cubeai::DeviceType::CPU);
					auto done_batch_t = torch_utils::TorchAdaptor::to_torch(done_batch, 
																			  cubeai::DeviceType::CPU);
					auto action_batch_t = torch_utils::TorchAdaptor::to_torch(action_batch, 
																			  cubeai::DeviceType::CPU);
															
					auto state_max = torch::max(q2, 1);
					auto state_max_val = std::get<0>(state_max);
					auto Y = reward_batch_t + GAMMA * ((1-done_batch_t) * state_max_val);
					auto X = q1.gather(1, action_batch_t.unsqueeze(1)).squeeze();
					auto loss = loss_fn(X, Y.detach());
					
					optimizer_ptr -> zero_grad();
					loss.backward();
					optimizer_ptr -> step();
					
					BOOST_LOG_TRIVIAL(info)<<"Loss at epoch: "<<loss.item<real_t>();
					epoch_loss.push_back(loss.item<real_t>());
					
				}
				
				step_counter += 1;
				
				// update done
				done = time_step.done();
				
				if(done || step_counter > TOTAL_ITRS_PER_EPOCH){ //#Q
					//done = true;
					BOOST_LOG_TRIVIAL(info)<<"Reward: "<<reward;
					BOOST_LOG_TRIVIAL(info)<<"Finishing epoch at step: "<<step_counter;
					step_counter = 0;
				}
            }

            BOOST_LOG_TRIVIAL(info)<<"Epoch finished...";
			
			// get the epsilon
			/*auto eps = policy.eps_value();
			if(eps > 0.1){
				eps -= 1/static_cast<real_t>(TOTAL_EPOCHS);
				policy.set_eps_value(eps);
			}*/
			
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
