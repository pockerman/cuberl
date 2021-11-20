#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/pg/vanilla_reinforce.h"
#include "cubeai/ml/distributions/torch_categorical.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/cart_pole.h"
#include "gymfcpp/time_step.h"

#include <torch/torch.h>

#include <deque>
#include <tuple>
#include <iostream>


namespace example{

namespace F = torch::nn::functional;

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::torch_tensor_t;
using cubeai::rl::algos::pg::Reinforce;
using cubeai::rl::algos::pg::ReinforceOpts;
using cubeai::ml::stats::TorchCategorical;
using gymfcpp::CartPole;


class PolicyImpl: public torch::nn::Module
{
public:


    PolicyImpl(const CartPole& env);

    torch_tensor_t forward(torch_tensor_t);

    template<typename StateTp>
    std::tuple<uint_t, torch_tensor_t> act(const StateTp& state);

    template<typename LossValuesTp>
    void update_policy_loss(const LossValuesTp& vals);

    void step_backward_policy_loss();

private:

   const CartPole* env_ptr_;
   torch::nn::Linear fc1_;
   torch::nn::Linear fc2_;


};


PolicyImpl::PolicyImpl(const CartPole& env)
    :
      env_ptr_(&env),
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
     torch::cat(torch::tensor(vals)).sum();
}


void
PolicyImpl::step_backward_policy_loss(){

}


torch_tensor_t
PolicyImpl::forward(torch_tensor_t x){

    x = F::relu(fc1_->forward(x));
    x = fc2_->forward(x);
    return F::softmax(x, F::SoftmaxFuncOptions(1));
}


template<typename StateTp>
std::tuple<uint_t, torch_tensor_t>
PolicyImpl::act(const StateTp& state){

    torch_tensor_t torch_state;

    auto probs = forward(torch_state);
    auto m = TorchCategorical(&probs, nullptr);
    auto action = m.sample();
    return std::make_tuple(action.item().to<uint_t>(), m.log_prob(action));

}

TORCH_MODULE(Policy);

}


int main(){

    using namespace example;

    try{

        Py_Initialize();
        auto gym_module = boost::python::import("gym");
        auto gym_namespace = gym_module.attr("__dict__");

        auto world = CartPole("v0", gym_namespace, false);
        world.make();

        Policy policy(world);
        torch::optim::Adam optimizer(policy->parameters(), torch::optim::AdamOptions(1e-2));
        ReinforceOpts opts = {1000, 100, 1.0e-2, 0.1};

        Reinforce<CartPole, Policy, torch::optim::Adam> reinforce(opts, world, policy, optimizer);
        reinforce.train();

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

    std::cout<<"This example requires PyTorch. Reconfigure cubeai with PyTorch support."<<std::endl;
    return 0;
}
#endif
