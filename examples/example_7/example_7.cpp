#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/policy_iteration.h"
#include "cubeai/rl/worlds/discrete_world.h"
#include "cubeai/rl/policies/uniform_discrete_policy.h"
#include "cubeai/rl/policies/stochastic_adaptor_policy.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/frozen_lake.h"
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
using cubeai::rl::policies::UniformDiscretePolicy;
using cubeai::rl::policies::StochasticAdaptorPolicy;
using cubeai::rl::algos::dp::PolicyIteration;

typedef gymfcpp::TimeStep<uint_t> time_step_t;


class FrozenLakeEnv: public DiscreteWorldBase<time_step_t>
{

public:

    typedef DiscreteWorldBase<time_step_t>::action_t action_t;
    typedef DiscreteWorldBase<time_step_t>::time_step_t time_step_t;

    //
    FrozenLakeEnv(gymfcpp::obj_t gym_namespace);

    ~FrozenLakeEnv() = default;

    virtual uint_t n_actions()const override final {return env_impl_.n_actions();}
    virtual uint_t n_states()const override final {return env_impl_.n_states();}
    virtual std::vector<std::tuple<real_t, uint_t, real_t, bool>> transition_dynamics(uint_t s, uint_t aidx)const override final;

    virtual time_step_t on_episode(const action_t&)override final {return time_step_t();}

    virtual time_step_t reset() override final;
    virtual  void build(bool reset) override final;
    virtual uint_t n_copies()const override final{return 1;}

private:

    // the environment implementation
    gymfcpp::FrozenLake env_impl_;

};

FrozenLakeEnv::FrozenLakeEnv(gymfcpp::obj_t gym_namespace)
    :
     DiscreteWorldBase<time_step_t>("FrozenLake"),
     env_impl_("v0", gym_namespace)
{}


FrozenLakeEnv::time_step_t
FrozenLakeEnv::reset(){
    return env_impl_.reset();
}

void
FrozenLakeEnv::build(bool reset){
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

    auto policy = std::make_shared<UniformDiscretePolicy>(env.n_states(), env.n_actions());

    std::cout<<"Policy before training..."<<std::endl;
    std::cout<<*policy<<std::endl;

    auto policy_adaptor = std::make_shared<StochasticAdaptorPolicy>(env.n_states(), env.n_actions(), policy);

    PolicyIteration<time_step_t> policy_itr(100, 1.0e-8, env, 1.0, 100,
                                                        policy, policy_adaptor);
    policy_itr.train();

    std::cout<<"Policy after training..."<<std::endl;
    std::cout<<*policy<<std::endl;

    // save the value function into a csv file
    policy_itr.save("policy_iteration.csv");

    return 0;
}


