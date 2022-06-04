#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/rl/algorithms/actor_critic/a2c.h"
#include "gymfcpp/cart_pole_env.h"
#include "gymfcpp/serial_vector_env_wrapper.h"
#include "gymfcpp/time_step.h"


#include <torch/torch.h>
#include <boost/python.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <tuple>

namespace{

using cubeai::rl::algos::ac::A2CConfig;
using cubeai::rl::algos::ac::A2C;
using rlenvs_cpp::gymfcpp::CartPole;
using rlenvs_cpp::obj_t;
using rlenvs_cpp::TimeStep;
using rlenvs_cpp::SerialVectorEnvWrapper;
using rlenvs_cpp::SerialVectorEnvWrapperConfig;

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::torch_tensor_t;

typedef TimeStep<torch_tensor_t> time_step_t;

typedef SerialVectorEnvWrapper<CartPole> env_type;

/*class DummyEnv
{

public:


    DummyEnv(obj_t gym_namespace);

    uint_t n_workers()const noexcept{return 1;}
    time_step_t step(const torch_tensor_t& action);
    time_step_t reset();


private:

    CartPole env_;


};

DummyEnv::DummyEnv(obj_t gym_namespace)
    :
      env_("v0", gym_namespace, true)

{}

time_step_t
DummyEnv::reset(){
    return time_step_t();
}

time_step_t
DummyEnv::step(const torch_tensor_t& action){
    return time_step_t();
}*/

class PolicyImpl: public torch::nn::Module
{
public:


    PolicyImpl();

    std::tuple<torch_tensor_t, torch_tensor_t> forward(torch_tensor_t);

    //template<typename StateTp>
    //std::tuple<uint_t, real_t> act(const StateTp& state);
    //
    //template<typename LossValuesTp>
    //void update_policy_loss(const LossValuesTp& vals);
    //
    //void step_backward_policy_loss();
    //
    //torch_tensor_t compute_loss(){return loss_;}

    torch_tensor_t sample(torch_tensor_t actions){}
    torch_tensor_t log_probabilities(torch_tensor_t actions){}

private:

   torch::nn::Linear fc1_;
   torch::nn::Linear fc2_;

   // placeholder for the loss
   torch_tensor_t loss_;

};

PolicyImpl::PolicyImpl()
    :
      fc1_(nullptr),
      fc2_(nullptr)
{}

std::tuple<torch_tensor_t, torch_tensor_t>
PolicyImpl::forward(torch_tensor_t){
    return std::make_tuple(torch_tensor_t(), torch_tensor_t());
}

TORCH_MODULE(Policy);

}


TEST(TestA2C, Test_Constructor) {

    Py_Initialize();
    auto gym_module = boost::python::import("__main__");
    auto gym_namespace = gym_module.attr("__dict__");

    //auto env = DummyEnv(gym_namespace);

    A2CConfig config;

    PolicyImpl policy;
    A2C<env_type, PolicyImpl> agent(config, policy);

}

TEST(TestA2C, Test_on_training_episode) {

    Py_Initialize();
    auto gym_module = boost::python::import("__main__");
    auto gym_namespace = gym_module.attr("__dict__");

    auto env_config = SerialVectorEnvWrapperConfig();
    env_config.env_id="v0";
    auto env = env_type(env_config, gym_namespace);

    A2CConfig config;

    PolicyImpl policy;
    A2C<env_type, PolicyImpl> agent(config, policy);
    agent.on_training_episode(env, static_cast<uint_t>(0));

}
#endif
