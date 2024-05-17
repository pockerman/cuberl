/**
 * This example illustrates how to use the REINFORCE algorithm
 * on the CartPole environment from Gymnasium
 *
 *
 *
 *
 * */
#include "cubeai/base/cubeai_config.h"

#if defined(USE_PYTORCH) && defined(USE_RLENVS_CPP)

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/pg/simple_reinforce.h"
#include "cubeai/rl/trainers/pytorch_rl_agent_trainer.h"
#include "cubeai/ml/distributions/torch_categorical.h"
#include "cubeai/optimization/optimizer_type.h"
#include "cubeai/optimization/pytorch_optimizer_factory.h"

#include "rlenvs/envs/gymnasium/classic_control/cart_pole_env.h"
#include <torch/torch.h>

#include <unordered_map>
#include <iostream>
#include <string>
#include <any>


namespace rl_example_13{


const std::string SERVER_URL = "http://0.0.0.0:8001/api";

namespace F = torch::nn::functional;

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::torch_tensor_t;
using cubeai::rl::algos::pg::SimpleReinforce;
using cubeai::rl::algos::pg::ReinforceConfig;
using cubeai::rl::PyTorchRLTrainer;
using cubeai::rl::PyTorchRLTrainerConfig;
using cubeai::ml::stats::TorchCategorical;
using rlenvs_cpp::envs::gymnasium::CartPole;


class PolicyImpl: public torch::nn::Module
{
public:


    PolicyImpl();

    torch_tensor_t forward(torch_tensor_t);

    template<typename StateTp>
    std::tuple<uint_t, real_t> act(const StateTp& state);

    template<typename LossValuesTp>
    void update_policy_loss(const LossValuesTp& vals);

    void step_backward_policy_loss();

    torch_tensor_t compute_loss(){return loss_;}

private:

   torch::nn::Linear fc1_;
   torch::nn::Linear fc2_;

   // placeholder for the loss
   torch_tensor_t loss_;

};


PolicyImpl::PolicyImpl()
    :
      fc1_(torch::nn::Linear(4, 16)),
      fc2_(torch::nn::Linear(16, 2))
{
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
}

template<typename LossValuesTp>
void
PolicyImpl::update_policy_loss(const LossValuesTp& vals){

     torch_tensor_t torch_vals = torch::tensor(vals);

     // specify that we require the gradient
     loss_ = torch::cat(torch::tensor(vals, torch::requires_grad())).sum();
}

void
PolicyImpl::step_backward_policy_loss(){
    loss_.backward();
}

torch_tensor_t
PolicyImpl::forward(torch_tensor_t x){

    x = F::relu(fc1_->forward(x));
    x = fc2_->forward(x);
    return F::softmax(x, F::SoftmaxFuncOptions(0));
}


template<typename StateTp>
std::tuple<uint_t, real_t>
PolicyImpl::act(const StateTp& state){

    auto torch_state = torch::tensor(state);

    auto probs = forward(torch_state);
    auto m = TorchCategorical(&probs, nullptr);
    auto action = m.sample();
    return std::make_tuple(action.item().toLong(), m.log_prob(action).item().to<real_t>());

}

TORCH_MODULE(Policy);

}


int main(){

    using namespace rl_example_13;

    try{



        auto env = CartPole(SERVER_URL);
         std::cout<<"Environment URL: "<<env.get_url()<<std::endl;
        std::unordered_map<std::string, std::any> options;

        std::cout<<"Creating the environment..."<<std::endl;
        env.make("v1", options);
        env.reset();
        std::cout<<"Done..."<<std::endl;

        std::cout<<"Number of actions="<<env.n_actions()<<std::endl;

        Policy policy;
        auto optimizer_ptr = std::make_unique<torch::optim::Adam>(policy->parameters(), torch::optim::AdamOptions(1e-2));

        // reinforce options
        ReinforceConfig opts = {1000, 100, 100, 100, 1.0e-2, 0.1, 195.0};
        SimpleReinforce<CartPole, Policy> algorithm(opts, policy);

        PyTorchRLTrainerConfig trainer_config{1.0e-8, 1001, 50};
        PyTorchRLTrainer<CartPole, SimpleReinforce<CartPole, Policy>> trainer(trainer_config, algorithm, std::move(optimizer_ptr));

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

    std::cout<<"This example requires PyTorch and gymfcpp. Reconfigure cuberl with USE_PYTORCH and USE_RLENVS_CPP flags turned ON."<<std::endl;
    return 0;
}
#endif
