#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/value_iteration.h"
#include "cubeai/rl/worlds/discrete_world.h"
#include "cubeai/rl/worlds/with_dynamics_mixin.h"
#include "cubeai/rl/policies/uniform_discrete_policy.h"
#include "cubeai/rl/policies/stochastic_adaptor_policy.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/frozen_lake_env.h"
#include "gymfcpp/time_step.h"

#include <boost/python.hpp>

#include <cmath>
#include <utility>
#include <tuple>
#include <iostream>
#include <random>
#include <algorithm>

namespace exe
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::envs::DiscreteWorldBase;
using cubeai::rl::envs::with_dynamics_mixin;
using cubeai::rl::policies::UniformDiscretePolicy;
using cubeai::rl::policies::StochasticAdaptorPolicy;
using cubeai::rl::algos::dp::ValueIteration;

typedef gymfcpp::TimeStep<uint_t> time_step_type;

class FrozenLakeEnv: public DiscreteWorldBase<time_step_type>, public with_dynamics_mixin
{

public:

    typedef DiscreteWorldBase<time_step_type>::action_type action_type;
    typedef DiscreteWorldBase<time_step_type>::time_step_type time_step_type;

    //
    FrozenLakeEnv(gymfcpp::obj_t gym_namespace);

    ~FrozenLakeEnv() = default;

    virtual uint_t n_actions()const override final {return env_impl_.n_actions();}
    virtual uint_t n_states()const override final {return env_impl_.n_states();}
    virtual std::vector<std::tuple<real_t, uint_t, real_t, bool>> transition_dynamics(uint_t s, uint_t aidx)const override final;

    virtual time_step_type step(const action_type&)override final {return time_step_type();}

    virtual time_step_type reset() override final;
    virtual  void build(bool reset) override final;
    virtual uint_t n_copies()const override final{return 1;}

private:

    // the environment implementation
    gymfcpp::FrozenLake env_impl_;

};

FrozenLakeEnv::FrozenLakeEnv(gymfcpp::obj_t gym_namespace)
    :
     DiscreteWorldBase<time_step_type>("FrozenLake"),
     env_impl_("v0", gym_namespace)
{}


FrozenLakeEnv::time_step_type
FrozenLakeEnv::reset(){
    return env_impl_.reset();
}

void
FrozenLakeEnv::build(bool /*reset*/){
    env_impl_.make();
}

std::vector<std::tuple<real_t, uint_t, real_t, bool>>
FrozenLakeEnv::transition_dynamics(uint_t s, uint_t aidx)const{
    return env_impl_.p(s, aidx);
}

}

int main() {

    using namespace exe;

    Py_Initialize();
    auto gym_module = boost::python::import("gym");
    auto gym_namespace = gym_module.attr("__dict__");

    FrozenLakeEnv env(gym_namespace);
    env.build(true);

    // start with a uniform random policy i.e.
    // the agnet knows nothing about the environment
    auto policy = std::make_shared<UniformDiscretePolicy>(env.n_states(), env.n_actions());

    std::cout<<"Policy before training..."<<std::endl;
    std::cout<<*policy<<std::endl;

    auto policy_adaptor = std::make_shared<StochasticAdaptorPolicy>(env.n_states(), env.n_actions(), policy);

    ValueIteration<time_step_type> value_itr(500, 1.0e-8, env, 1.0, policy, policy_adaptor);
    value_itr.do_verbose_output();
    value_itr.train();

    std::cout<<"Optimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):"<<std::endl;
    std::cout<<*policy<<std::endl;

    // save the value function into a csv file
    value_itr.save("value_iteration.csv");

    return 0;
}


