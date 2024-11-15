# Example 11: An A2C Solver for CartPole

Example <a href="https://github.com/pockerman/cuberl/blob/master/examples/rl/rl_example_13/rl_example_13.md">Example 13: REINFORCE algorithm on CartPole</a>
introduce the vanilla REINFORCE algorithm in order to solve the pole balancing problem. 
The REINFORCE algorithm is a simple algorithm and easy to use however it exhibits a large variance in the reward signal.

In this exampel, we will introduce <a href="https://proceedings.neurips.cc/paper_files/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf">actor-critic</a> methods
and specifically the A2C algorithm.	The main objective of actor-critic methods is to further reduce the high gradient variance. 
One way towards this direction is to use the so-called reward-to-go term for a trajectory $T$. 

$$G = \sum_{k = t}^T R(s_{k}, \alpha_{k})$$

For this example to work, you will need to have both PyTorch and ```rlenvs_cpp``` enabled.


The workings of the A2C algorithm are handled by the <a href="https://github.com/pockerman/cuberl/blob/master/include/cubeai/rl/algorithms/actor_critic/a2c.h">A2CSolver</a>
class. The class accepts three template parameters

- The environment type 
- The policy type or the actor type 
- The critic type

The role of the actor is to select an action for the agent to take.
The role of the critic is to tell us whether that action was good or bad.
We could use of course the raw reward signal to have an assessment on this
but for enviroments where the reward is sparse this may not work or for enviroments
where the reward signal is the same for most actions.

Given this approach, the critic will be a term in the actor’s loss function,
whilst the critic will learn directly from the provided reward signals.

In the code below, the critic and action networks do not share any details.
This need not be the case however. You can come up with an implementation
where the two networks share most of the layers and only differentiate
at the output layer.


### The Actor network class

```
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
```


## The driver code

The driver code for this tutorial is shown below. 


```cpp
#include "cubeai/base/cubeai_config.h"

#if defined(USE_PYTORCH) && defined(USE_RLENVS_CPP)

#include "cubeai/base/cubeai_types.h"
#include "cubeai/maths/statistics/distributions/torch_categorical.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/algorithms/actor_critic/a2c.h"
#include "cubeai/maths/optimization/optimizer_type.h"
#include "cubeai/maths/optimization/pytorch_optimizer_factory.h"
#include "rlenvs/envs/gymnasium/classic_control/cart_pole_env.h"

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
   linear2_ = register_module("linea2_", torch::nn::Linear(128, 256));
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

    using namespace rl_example_11;

    try{

        // create the environment
        env_type env(SERVER_URL);

        std::cout<<"Environment URL: "<<env.get_url()<<std::endl;
        std::unordered_map<std::string, std::any> options;

        std::cout<<"Creating the environment..."<<std::endl;
        env.make("v1", options);
        env.reset();
        std::cout<<"Done..."<<std::endl;
        std::cout<<"Number of actions="<<env.n_actions()<<std::endl;


        A2CConfig a2c_config;
        ActorNet policy(4, env.n_actions());
        CriticNet critic(4);


        std::map<std::string, std::any> opt_options;
        opt_options.insert(std::make_pair("lr", 0.001));

        auto pytorch_ops = cubeai::maths::optim::pytorch::build_pytorch_optimizer_options(cubeai::maths::optim::OptimzerType::ADAM,
                                                                                          opt_options);

        auto policy_optimizer = cubeai::maths::optim::pytorch::build_pytorch_optimizer(cubeai::maths::optim::OptimzerType::ADAM,
                                                                                       *policy, pytorch_ops);

        auto critic_optimizer = cubeai::maths::optim::pytorch::build_pytorch_optimizer(cubeai::maths::optim::OptimzerType::ADAM,
                                                                                       *critic, pytorch_ops);

        typedef A2CSolver<env_type, ActorNet, CriticNet> solver_type;

        solver_type solver(a2c_config, policy, critic,
                           policy_optimizer, critic_optimizer);

        RLSerialTrainerConfig config;
        RLSerialAgentTrainer<env_type, solver_type> trainer(config, solver);
        trainer.train(env);

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
```

Running the code above produces the following output

```

```

