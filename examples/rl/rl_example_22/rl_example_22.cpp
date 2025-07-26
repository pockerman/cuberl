#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"

#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/algorithms/pg/ppo.h"
#include "cubeai/rl/algorithms/pg/ppo_config.h"
#include "cubeai/maths/optimization/optimizer_type.h"
#include "cubeai/maths/statistics/distributions/torch_normal.h"
#include "cubeai/maths/optimization/pytorch_optimizer_factory.h"
#include "cubeai/utils/torch_adaptor.h"

#include "rlenvs/utils/io/csv_file_writer.h"
#include "rlenvs/envs/api_server/apiserver.h"
#include "rlenvs/envs/gymnasium/box2d/lunar_lander_env.h"

#include <boost/log/trivial.hpp>
#include <torch/torch.h>

#include <vector>
#include <iostream>
#include <unordered_map>
#include <filesystem>

namespace rl_example_11{

const std::string SERVER_URL = "http://0.0.0.0:8001/api";
const std::string EXPERIMENT_ID = "1";
const std::string POLICY = "policy.csv";

namespace F = torch::nn::functional;

using cuberl::real_t;
using cuberl::uint_t;
using cuberl::torch_tensor_t;
using cuberl::DeviceType;
using cuberl::maths::stats::TorchNormalDist;
using cuberl::rl::algos::pg::PPOConfig;
using cuberl::rl::algos::pg::PPOSolver;
using cuberl::rl::RLSerialAgentTrainer;
using cuberl::rl::RLSerialTrainerConfig;
using cuberl::utils::pytorch::TorchAdaptor;
using rlenvscpp::envs::RESTApiServerWrapper;

typedef  rlenvscpp::envs::gymnasium::LunarLanderContinuousEnv env_type;

/// Layer sizes for Actor
const uint_t L1 = 4;
const uint_t L2 = 128;
const uint_t L3 = 2;
const real_t LEARNING_RATE = 0.01;
const auto N_EPISODES = 2000;


// create the Action network
class ActorNetImpl: public torch::nn::Module
{
public:

    
    ActorNetImpl();
	 std::pair<torch::Tensor, torch::Tensor> forward(torch_tensor_t state);
	template<typename StateTp>
	std::tuple<std::vector<real_t>, torch_tensor_t> act(const StateTp& state);

    /// Evaluate the actor
    /// @param states //
    /// @param actions
    /// @return
    std::tuple<torch_tensor_t, torch_tensor_t, torch_tensor_t>
	evaluate(torch_tensor_t states, torch_tensor_t actions);
			   
private:

    torch::nn::Linear fc1_;
	torch::nn::Dropout dp_;
    torch::nn::Linear  fc2_;
	torch::nn::Linear  mean_layer_;
	torch::nn::Linear log_std_;

};

ActorNetImpl::ActorNetImpl()
:
torch::nn::Module(),
fc1_(torch::nn::Linear(L1, L2)),
dp_(torch::nn::Dropout(0.6)),
fc2_(torch::nn::Linear(L2, L3)),
mean_layer_(torch::nn::Linear(64, 2)),
log_std_(torch::nn::Linear(64, 2))
{
   register_module("fc1_", fc1_);
   register_module("dp_",  dp_);
   register_module("fc2_", fc2_);
   register_module("mean_layer_", mean_layer_);
	register_module("log_std_", log_std_);
}

template<typename StateTp>
std::tuple<std::vector<real_t>, torch_tensor_t>
ActorNetImpl::act(const StateTp& state){

	auto torch_state = torch::tensor(state);
	auto [mean, sd_dev] = this->forward(torch_state);
	auto dist = TorchNormalDist(mean, sd_dev);

	// sample an action
	auto action = dist.sample();
	auto log_prob = dist.log_prob(action).sum();

	auto clamped_action = TorchAdaptor::to_vector<real_t>(torch::clamp(action, -1.0, 1.0));
	return {clamped_action, log_prob};
}

std::pair<torch_tensor_t, torch_tensor_t>
ActorNetImpl::forward(torch_tensor_t state){
	
	state = torch::relu(fc1_ -> forward(state));
	state = torch::relu(fc2_ -> forward(state));
	auto mean = mean_layer_->forward(state);
	auto log_std = log_std_->forward(state);
	auto std = torch::exp(log_std);
	return {mean, std};
}

std::tuple<torch_tensor_t, torch_tensor_t, torch_tensor_t>
ActorNetImpl::evaluate(torch_tensor_t states, torch_tensor_t actions) {
	auto [mean, std] = this->forward(states);
	auto dist = TorchNormalDist(mean, std);
	auto log_probs = dist.log_prob(actions).sum(1);
	auto entropy = dist.entropy().sum(1).mean();
	return {log_probs, entropy, mean};
}


/// The Critic network 
class CriticNetImpl: public torch::nn::Module
{
public:

    CriticNetImpl();
    torch_tensor_t forward(torch_tensor_t state);
	
	template<typename StateTp>
	torch_tensor_t evaluate(const StateTp& state); 

private:

    torch::nn::Linear fc1_;
    torch::nn::Linear fc2_;
    torch::nn::Linear fc3_;

};


CriticNetImpl::CriticNetImpl()
:
torch::nn::Module(),
fc1_(torch::nn::Linear(4, 128)),
fc2_(torch::nn::Linear(128, 256)),
fc3_(torch::nn::Linear(256, 1))
{
   register_module("fc1_", fc1_);
   register_module("fc2_", fc2_);
   register_module("fc3_", fc3_);
}

torch_tensor_t
CriticNetImpl::forward(torch_tensor_t state){

    auto output = F::relu(fc1_ -> forward(state));
    output = F::relu(fc2_ -> forward(output));
    output = fc3_ -> forward(output);
    return output;
}

template<typename StateTp>
torch_tensor_t 
CriticNetImpl::evaluate(const StateTp& state){
	
	auto torch_state = torch::tensor(state);
	return forward(torch_state);
}

TORCH_MODULE(ActorNet);
TORCH_MODULE(CriticNet);

}


int main(){

	BOOST_LOG_TRIVIAL(info)<<"Starting agent training";
    using namespace rl_example_11;

    try{
		
		// let's create a directory where we want to
        //store all the results from running a simualtion
        std::filesystem::create_directories("experiments/" + EXPERIMENT_ID);

		torch::manual_seed(42);
		RESTApiServerWrapper server(SERVER_URL, true);
		
        // create the environment
        env_type env(server);

		BOOST_LOG_TRIVIAL(info)<<"Creating environment...";
        std::unordered_map<std::string, std::any> options;

        env.make("v1", options);
        env.reset();
		
        BOOST_LOG_TRIVIAL(info)<<"Done...";
		BOOST_LOG_TRIVIAL(info)<<"Number of actions="<<env.n_actions();
        
        PPOConfig ppo_config;
		ppo_config.max_itrs_per_episode = 500; 
		ppo_config.n_episodes = N_EPISODES;
		ppo_config.device_type = DeviceType::CPU;
        
		ActorNet policy;
        CriticNet critic;

        std::map<std::string, std::any> opt_options;
        opt_options.insert(std::make_pair("lr", LEARNING_RATE));

		using namespace cuberl::maths::optim::pytorch;
		using namespace cuberl::maths::optim;

        auto pytorch_ops = build_pytorch_optimizer_options(OptimzerType::ADAM,
														  opt_options);

        auto policy_optimizer = build_pytorch_optimizer(OptimzerType::ADAM,
                                                        *policy, pytorch_ops);

        auto critic_optimizer = build_pytorch_optimizer(OptimzerType::ADAM,
                                                        *critic, pytorch_ops);

        typedef PPOSolver<env_type, ActorNet, CriticNet> solver_type;

        solver_type solver(ppo_config, policy, critic,
                           policy_optimizer, critic_optimizer);

        RLSerialTrainerConfig config;
		config.n_episodes = N_EPISODES;
		config.output_msg_frequency = 20;
        
		RLSerialAgentTrainer<env_type, solver_type> trainer(config, solver);
        
		auto info = trainer.train(env);
		
		BOOST_LOG_TRIVIAL(info)<<"Training info...";
		BOOST_LOG_TRIVIAL(info)<<info;
		
		auto experiment_path = std::string("experiments/") + EXPERIMENT_ID;
		
		// write the loss values
		auto& policy_loss_vals= solver.get_monitor().policy_loss_values;
		
		rlenvscpp::utils::io::CSVWriter loss_csv_writer(experiment_path + "/" + "policy_loss.csv",
														  rlenvscpp::utils::io::CSVWriter::default_delimiter());
		loss_csv_writer.open();
		
		auto episode_counter = 0;
		for(uint_t i=0; i<policy_loss_vals.size(); ++i){
			std::tuple<uint_t, real_t> row = {episode_counter++, policy_loss_vals[i]};
			loss_csv_writer.write_row(row);
		}
		
		loss_csv_writer.close();
		
		// write the loss values
		auto& critic_loss_vals  = solver.get_monitor().critic_loss_values;
		
		rlenvscpp::utils::io::CSVWriter critic_csv_writer(experiment_path + "/" + "critic_loss.csv",
														  rlenvscpp::utils::io::CSVWriter::default_delimiter());
		critic_csv_writer.open();
		
		episode_counter = 0;
		for(uint_t i=0; i<policy_loss_vals.size(); ++i){
			std::tuple<uint_t, real_t> row = {episode_counter++, critic_loss_vals[i]};
			critic_csv_writer.write_row(row);
		}
		
		critic_csv_writer.close();

		auto& rewards  = solver.get_monitor().rewards;
		rlenvscpp::utils::io::CSVWriter rewards_csv_writer(experiment_path + "/" + "rewards.csv",
														  rlenvscpp::utils::io::CSVWriter::default_delimiter());
		rewards_csv_writer.open();
		
		episode_counter = 0;
		for(uint_t i=0; i<rewards.size(); ++i){
			std::tuple<uint_t, real_t> row = {episode_counter++, rewards[i]};
			rewards_csv_writer.write_row(row);
		}
		
		rewards_csv_writer.close();
		
		
		auto& episode_duration  = solver.get_monitor().episode_duration;
		rlenvscpp::utils::io::CSVWriter episode_duration_csv_writer(experiment_path + "/" + "episode_duration.csv",
														            rlenvscpp::utils::io::CSVWriter::default_delimiter());
		episode_duration_csv_writer.open();
		
		episode_counter = 0;
		for(uint_t i=0; i<episode_duration.size(); ++i){
			std::tuple<uint_t, real_t> row = {episode_counter++, episode_duration[i]};
			episode_duration_csv_writer.write_row(row);
		}
		
		episode_duration_csv_writer.close();
		
		BOOST_LOG_TRIVIAL(info)<<"Finished agent training";

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

    std::cout<<"This example requires PyTorch. Reconfigure cuberl with USE_PYTORCH fag turned ON."<<std::endl;
    return 1;
}
#endif


/*

#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <cmath>

struct ActorImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, mean_layer{nullptr}, log_std_layer{nullptr};

    ActorImpl(int input_dim, int action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        mean_layer = register_module("mean_layer", torch::nn::Linear(64, action_dim));
        log_std_layer = register_module("log_std_layer", torch::nn::Linear(64, action_dim));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        auto mean = mean_layer->forward(x);
        auto log_std = log_std_layer->forward(x);
        auto std = torch::exp(log_std);
        return {mean, std};
    }

    template<typename StateTp>
    std::tuple<torch::Tensor, torch::Tensor> act(const StateTp& state) {
        auto torch_state = torch::tensor(state);
        auto [mean, std] = this->forward(torch_state);
        auto dist = torch::distributions::Normal(mean, std);
        auto action = dist.sample();
        auto log_prob = dist.log_prob(action).sum();
        return {torch::clamp(action, -1.0, 1.0), log_prob};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(torch::Tensor states, torch::Tensor actions) {
        auto [mean, std] = this->forward(states);
        auto dist = torch::distributions::Normal(mean, std);
        auto log_probs = dist.log_prob(actions).sum(1);
        auto entropy = dist.entropy().sum(1).mean();
        return {log_probs, entropy, mean};
    }
};
TORCH_MODULE(Actor);

struct CriticImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, output{nullptr};

    CriticImpl(int input_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        output = register_module("output", torch::nn::Linear(64, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return output->forward(x);
    }
};
TORCH_MODULE(Critic);

class PPOAgent {
public:
    PPOAgent(int state_dim, int action_dim)
        : actor(state_dim, action_dim),
          critic(state_dim),
          actor_optimizer(actor->parameters(), torch::optim::AdamOptions(1e-3)),
          critic_optimizer(critic->parameters(), torch::optim::AdamOptions(1e-3)) {
        actor->to(torch::kCPU);
        critic->to(torch::kCPU);
    }

    torch::Tensor select_action(torch::Tensor state, torch::Tensor& log_prob) {
        auto [action, logp] = actor->act(state);
        log_prob = logp;
        return action;
    }

    torch::Tensor evaluate_value(torch::Tensor state) {
        return critic->forward(state);
    }

    void store_transition(torch::Tensor state, torch::Tensor action, torch::Tensor reward,
                          torch::Tensor log_prob, torch::Tensor value) {
        states.push_back(state);
        actions.push_back(action);
        rewards.push_back(reward);
        log_probs.push_back(log_prob);
        values.push_back(value);
    }

    void update() {
        auto states_tensor = torch::stack(states);
        auto actions_tensor = torch::stack(actions);
        auto rewards_tensor = torch::stack(rewards);
        auto old_log_probs_tensor = torch::stack(log_probs);
        auto values_tensor = torch::stack(values).detach();

        std::vector<torch::Tensor> returns;
        torch::Tensor R = torch::zeros(1);
        for (int i = rewards.size() - 1; i >= 0; --i) {
            R = rewards[i] + gamma * R;
            returns.insert(returns.begin(), R);
        }
        auto returns_tensor = torch::stack(returns);
        auto advantages = returns_tensor - values_tensor;

        for (int epoch = 0; epoch < 4; ++epoch) {
            auto [new_log_probs, entropy, _] = actor->evaluate(states_tensor, actions_tensor);

            auto ratio = (new_log_probs - old_log_probs_tensor).exp();
            auto surr1 = ratio * advantages;
            auto surr2 = torch::clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages;
            auto actor_loss = -torch::min(surr1, surr2).mean();

            auto value_estimates = critic->forward(states_tensor).squeeze();
            auto critic_loss = torch::nn::functional::mse_loss(value_estimates, returns_tensor);

            auto total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy;

            actor_optimizer.zero_grad();
            critic_optimizer.zero_grad();
            total_loss.backward();
            actor_optimizer.step();
            critic_optimizer.step();
        }

        states.clear();
        actions.clear();
        rewards.clear();
        log_probs.clear();
        values.clear();
    }

private:
    Actor actor;
    Critic critic;
    torch::optim::Adam actor_optimizer;
    torch::optim::Adam critic_optimizer;

    std::vector<torch::Tensor> states, actions, rewards, log_probs, values;
    double gamma = 0.99;
    double clip_epsilon = 0.2;
};

*/