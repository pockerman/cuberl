# Example 13: REINFORCE algorithm on CartPole

The DQN algorithm we used in examples <a href="../rl_example_12/rl_example_12.md">EXample 12: DQN algorithm on Gridworld</a>
and <a href="../rl_example_15/rl_example_15.md">EXample 15: DQN algorithm on Gridworld with experience replay</a> approximate a value function
and in particular the Q state-action value function. 
However, in reinforcement learning we are more interested in policies since a policy dictates how 
an agent behaves in a given state.

In this example, we will implement a policy-based method. In particular, we will use one of the earliest policy-based methods namely the
REINFORCE algorithm. Policy-based methods approximate policies directly. 
In order to do this, we will use the ```gymnasium.CartPole``` environment.
Specifically, we will use the class <a href="https://github.com/pockerman/rlenvs_from_cpp/blob/master/src/rlenvs/envs/gymnasium/classic_control/cart_pole_env.h">CartPole</a> class from
<a href="https://github.com/pockerman/rlenvs_from_cpp/tree/master">rlenvs_from_cpp</a> library. This is a simple class that allows us to
sent HTTP requests to a server that actually runs the environment. We need this as ```gymnasium.CartPole``` is written in Python and we 
want to avoid the complexity of directly executing Python code from the C++ driver.

We will use the policy network from the book <a href="https://www.manning.com/books/deep-reinforcement-learning-in-action">Deep Reinforcement Learning in Action</a>
by Manning Publications. However, feel free to experiment with this. 

The REINFORCE algorithm is implemented in the class <a href="https://github.com/pockerman/cuberl/blob/master/include/cubeai/rl/algorithms/pg/simple_reinforce.h">ReinforceSolver</a>
The solver is passed to the ```RLSerialAgentTrainer``` class that manages the loop over the specified number of episodes.
The ```ReinforceSolver``` class overrides some virtual methods defined in the ```RLSolverBase``` class.

The class ```ReinforceSolver``` accepts three template parameters:

- The environment type 
- The policy type 
- The loss function type 

The environment type is a standard argument for all RL solvers in ```cubeai```. The policy type represents the 
PyTorch model that we will use, whilst the loss function type  represents the object responsible for calculating the 
model loss during training. 

Below is the code for the network that implements the policy we want to use

## The ```PolicyImpl``` class 

```
class PolicyNetImpl: public torch::nn::Module
{
public:


    PolicyNetImpl();

    torch_tensor_t forward(torch_tensor_t state);

    template<typename StateTp>
    std::tuple<uint_t, real_t> act(const StateTp& state);
	
private:

   torch::nn::Linear fc1_;
   torch::nn::Linear fc2_;

};


PolicyNetImpl::PolicyNetImpl()
    :
      fc1_(torch::nn::Linear(4, 16)),
      fc2_(torch::nn::Linear(16, 2))
{
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
}


torch_tensor_t
PolicyNetImpl::forward(torch_tensor_t x){

    x = F::relu(fc1_->forward(x));
    x = fc2_->forward(x);
    return F::softmax(x, F::SoftmaxFuncOptions(0));
}


template<typename StateTp>
std::tuple<uint_t, real_t>
PolicyNetImpl::act(const StateTp& state){

    auto torch_state = torch::tensor(state);

    auto probs = forward(torch_state);
    auto m = TorchCategorical(probs, false);
    auto action = m.sample();
    return std::make_tuple(action.item().toLong(), 
	                       m.log_prob(action).item().to<real_t>());

}
```

In ```ReinforceSolver``` expects a policy type that exposes an ```act``` function that
returns the action type and log probability of taking this action

## Summary

This example introduced the ```ReinforceSolver``` class that models the REINFORCE algorithm.

The RINFORCE algorithm, just like all policy gradient methods, suffers from high variance in the gradient estimation.
There are several reasons behind this high variance; e.g. sparse rewards or environment randomness.
Regardless of reasons behind high variance in the gradient, its effect can be detrimental during learning as it destabilizes it. Hence, reducing the high variance is important for feasible training. 


## Driver code

```/**
 * This example illustrates how to use the REINFORCE algorithm
 * on the CartPole environment from Gymnasium
 **/
#include "cubeai/base/cubeai_config.h"

#if defined(USE_PYTORCH) && defined(USE_RLENVS_CPP)

#include "cubeai/base/cubeai_types.h"
#include "cubeai/io/csv_file_writer.h"
#include "cubeai/rl/algorithms/pg/simple_reinforce.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/maths/optimization/optimizer_type.h"
#include "cubeai/maths/optimization/pytorch_optimizer_factory.h"
#include "cubeai/maths/statistics/distributions/torch_categorical.h"

#include "rlenvs/envs/gymnasium/classic_control/cart_pole_env.h"
#include <torch/torch.h>

#include <unordered_map>
#include <iostream>
#include <string>
#include <any>
#include <filesystem>
#include <map>

namespace rl_example_13{


const std::string SERVER_URL = "http://0.0.0.0:8001/api";
const std::string EXPERIMENT_ID = "2";

namespace F = torch::nn::functional;

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::torch_tensor_t;
using cubeai::DeviceType;
using cubeai::rl::algos::pg::ReinforceSolver;
using cubeai::rl::algos::pg::ReinforceConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using cubeai::maths::stats::TorchCategorical;
using rlenvs_cpp::envs::gymnasium::CartPole;


const uint_t L1 = 4;
const uint_t L2 = 150;
const uint_t L3 = 3;
const real_t LEARNING_RATE = 0.0009;


// The class that models the Policy network to train
class PolicyNetImpl: public torch::nn::Module
{
public:

    PolicyNetImpl();
    torch_tensor_t forward(torch_tensor_t state);

    template<typename StateTp>
    std::tuple<uint_t, real_t> act(const StateTp& state);
	
private:

   torch::nn::Linear fc1_;
   torch::nn::Linear fc2_;
};


PolicyNetImpl::PolicyNetImpl()
    :
      fc1_(torch::nn::Linear(4, 16)),
      fc2_(torch::nn::Linear(16, 2))
{
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
}


torch_tensor_t
PolicyNetImpl::forward(torch_tensor_t x){

    x = F::relu(fc1_->forward(x));
    x = fc2_->forward(x);
    return F::softmax(x, F::SoftmaxFuncOptions(0));
}


template<typename StateTp>
std::tuple<uint_t, real_t>
PolicyNetImpl::act(const StateTp& state){

    auto torch_state = torch::tensor(state);

    auto probs = forward(torch_state);
    auto m = TorchCategorical(probs, false);
    auto action = m.sample();
    return std::make_tuple(action.item().toLong(), 
	                       m.log_prob(action).item().to<real_t>());

}

TORCH_MODULE(PolicyNet);


struct Loss_1
{
	torch_tensor_t operator()(torch_tensor_t preds, torch_tensor_t y)const;
};

torch_tensor_t 
Loss_1::operator()(torch_tensor_t preds, torch_tensor_t y)const{
	return -1.0 * torch::sum(y * torch::log(preds));
}

typedef Loss_1 loss_type;
typedef CartPole env_type;
typedef PolicyNet policy_type;
typedef ReinforceSolver<env_type, PolicyNet, loss_type> solver_type;
}


int main(){

    using namespace rl_example_13;

    try{

        // let's create a directory where we want to
        //store all the results from running a simualtion
        std::filesystem::create_directories("experiments/" + EXPERIMENT_ID);
        torch::manual_seed(42);

        auto env = CartPole(SERVER_URL);
        std::cout<<"Environment URL: "<<env.get_url()<<std::endl;

        std::cout<<"Creating the environment..."<<std::endl;
        std::unordered_map<std::string, std::any> options;

        // with Gymnasium v0 is not working
        env.make("v1", options);
        env.reset();

        std::cout<<"Done..."<<std::endl;
        std::cout<<"Number of actions="<<env.n_actions()<<std::endl;

        PolicyNet policy;

        
        // reinforce options
        ReinforceConfig opts = {true, 1000, 100, 100, 
		                        100, 1.0e-2, 0.1, 195.0,
								DeviceType::CPU};

        std::map<std::string, std::any> opt_options;
        opt_options.insert(std::make_pair("lr", LEARNING_RATE));

		using namespace cubeai::maths;
        auto pytorch_ops = optim::pytorch::build_pytorch_optimizer_options(optim::OptimzerType::ADAM,
																		   opt_options);

        auto policy_optimizer = optim::pytorch::build_pytorch_optimizer(optim::OptimzerType::ADAM,
																		*policy, pytorch_ops);

		loss_type loss;
        solver_type solver(opts, policy, loss, policy_optimizer);


        RLSerialTrainerConfig config;
        config.n_episodes = 10;
        config.output_msg_frequency = 10;
        RLSerialAgentTrainer<env_type, solver_type> trainer(config, solver);
        trainer.train(env);

        auto info = trainer.train(env);
        std::cout<<"Trainer info: "<<info<<std::endl;

        // save the rewards per episode for visualization
        // purposes
        auto filename = std::string("experiments/") + EXPERIMENT_ID;
        filename += "/reinforce_rewards.csv";
        cubeai::io::CSVWriter csv_writer(filename, cubeai::io::CSVWriter::default_delimiter());
        csv_writer.open();
        csv_writer.write_column_vector(trainer.episodes_total_rewards());


        // save the policy also so that we can load it and check
        // use it
        auto policy_model_filename = std::string("experiments/") + 
		                             EXPERIMENT_ID + std::string("/reinforce_cartpole_policy.pth");
        
        torch::serialize::OutputArchive archive;
        policy->save(archive);
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

    std::cout<<"This example requires PyTorch and gymfcpp. Reconfigure cuberl with USE_PYTORCH and USE_RLENVS_CPP flags turned ON."<<std::endl;
    return 0;
}
#endif

```