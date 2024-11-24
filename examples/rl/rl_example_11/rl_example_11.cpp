#include "cubeai/base/cubeai_config.h"

#if defined(USE_PYTORCH) && defined(USE_RLENVS_CPP)

#include "cubeai/base/cubeai_types.h"
#include "cubeai/maths/statistics/distributions/torch_categorical.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/algorithms/actor_critic/a2c.h"
#include "cubeai/maths/optimization/optimizer_type.h"
#include "cubeai/maths/optimization/pytorch_optimizer_factory.h"
#include "rlenvs/envs/gymnasium/classic_control/cart_pole_env.h"

#include <boost/log/trivial.hpp>

#include <iostream>
#include <iostream>
#include <unordered_map>

namespace rl_example_11{

const std::string SERVER_URL = "http://0.0.0.0:8001/api";

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::torch_tensor_t;
using cubeai::maths::stats::TorchCategorical;
using cubeai::rl::algos::ac::A2CConfig;
using cubeai::rl::algos::ac::A2CSolver;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using rlenvs_cpp::envs::gymnasium::CartPoleActionsEnum;


// create the Action and the Critic networks
class ActorNetImpl: public torch::nn::Module
{
public:

    // constructor
    ActorNetImpl(uint_t state_size, uint_t action_size);


    torch_tensor_t forward(torch_tensor_t state);
    torch_tensor_t log_probabilities(torch_tensor_t actions);
    torch_tensor_t sample();


private:

    torch::nn::Linear linear1_;
    torch::nn::Linear linear2_;
    torch::nn::Linear linear3_;

    // the underlying distribution used to sample actions
    TorchCategorical distribution_;
};

ActorNetImpl::ActorNetImpl(uint_t state_size, uint_t action_size)
:
torch::nn::Module(),
linear1_(nullptr),
linear2_(nullptr),
linear3_(nullptr)
{
   linear1_ = register_module("linear1_", torch::nn::Linear(state_size, 128));
   linear2_ = register_module("linea2_", torch::nn::Linear(128, 256));
   linear3_ = register_module("linear3_", torch::nn::Linear(256, action_size));
}


torch_tensor_t
ActorNetImpl::forward(torch_tensor_t state){

    auto output = torch::nn::functional::relu(linear1_(state));
    output = torch::nn::functional::relu(linear2_(output));
    output = linear3_(output);
    const torch_tensor_t probs = torch::nn::functional::softmax(output,-1);
    distribution_.build_from_probabilities(probs);
    return probs;
}

torch_tensor_t
ActorNetImpl::sample(){
    return distribution_.sample();
}

torch_tensor_t
ActorNetImpl::log_probabilities(torch_tensor_t actions){
    return distribution_.log_prob(actions);
}


class CriticNetImpl: public torch::nn::Module
{
public:

    // constructor
    CriticNetImpl(uint_t state_size);

    torch_tensor_t forward(torch_tensor_t state);

private:

    torch::nn::Linear linear1_;
    torch::nn::Linear linear2_;
    torch::nn::Linear linear3_;

};


CriticNetImpl::CriticNetImpl(uint_t state_size)
:
torch::nn::Module(),
linear1_(nullptr),
linear2_(nullptr),
linear3_(nullptr)
{
   linear1_ = register_module("linear1_", torch::nn::Linear(state_size, 128));
   linear2_ = register_module("linear2_", torch::nn::Linear(128, 256));
   linear3_ = register_module("linear3_", torch::nn::Linear(256, 1));
}

torch_tensor_t
CriticNetImpl::forward(torch_tensor_t state){

    auto output = torch::nn::functional::relu(linear1_(state));
    output = torch::nn::functional::relu(linear2_(output));
    output = linear3_(output);
    return output;
}

TORCH_MODULE(ActorNet);
TORCH_MODULE(CriticNet);

typedef  rlenvs_cpp::envs::gymnasium::CartPole env_type;

}


int main(){

	BOOST_LOG_TRIVIAL(info)<<"Starting agent training";
    using namespace rl_example_11;

    try{

        // create the environment
        env_type env(SERVER_URL);

		BOOST_LOG_TRIVIAL(info)<<"Creating environment...";
        std::unordered_map<std::string, std::any> options;

        env.make("v1", options);
        env.reset();
        BOOST_LOG_TRIVIAL(info)<<"Done...";
		BOOST_LOG_TRIVIAL(info)<<"Number of actions="<<env.n_actions();
        
        A2CConfig a2c_config;
		a2c_config.n_iterations_per_episode = 20;
        ActorNet policy(4, env.n_actions());
        CriticNet critic(4);


        std::map<std::string, std::any> opt_options;
        opt_options.insert(std::make_pair("lr", 0.001));


		using namespace cubeai::maths::optim::pytorch;

        auto pytorch_ops = build_pytorch_optimizer_options(cubeai::maths::optim::OptimzerType::ADAM,
                                                                                          opt_options);

        auto policy_optimizer = build_pytorch_optimizer(cubeai::maths::optim::OptimzerType::ADAM,
                                                                                       *policy, pytorch_ops);

        auto critic_optimizer = build_pytorch_optimizer(cubeai::maths::optim::OptimzerType::ADAM,
                                                                                       *critic, pytorch_ops);

        typedef A2CSolver<env_type, ActorNet, CriticNet> solver_type;

        solver_type solver(a2c_config, policy, critic,
                           policy_optimizer, critic_optimizer);

        RLSerialTrainerConfig config;
		config.n_episodes = 100;
        RLSerialAgentTrainer<env_type, solver_type> trainer(config, solver);
        auto info = trainer.train(env);
		
		BOOST_LOG_TRIVIAL(info)<<"Training info...";
		BOOST_LOG_TRIVIAL(info)<<info;
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

    std::cout<<"This example requires the flag USE_RLENVS_CPP to be true."<<std::endl;
    std::cout<<"Reconfigures and rebuild the library by setting the flag USE_RLENVS_CPP  to ON."<<std::endl;
    return 1;
}
#endif
