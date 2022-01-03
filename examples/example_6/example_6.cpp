#include "cubeai/base/cubeai_config.h"

#ifdef USE_GYMFCPP

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/iterative_policy_evaluation.h"
#include "cubeai/rl/worlds/discrete_world.h"
#include "cubeai/rl/worlds/with_dynamics_mixin.h"
#include "cubeai/rl/policies/uniform_discrete_policy.h"

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
using cubeai::rl::envs::with_dynamics_mixin;
using cubeai::rl::policies::UniformDiscretePolicy;
using cubeai::rl::algos::dp::IterativePolicyEval;

typedef gymfcpp::TimeStep<uint_t> time_step_t;


class FrozenLakeEnv: public DiscreteWorldBase<gymfcpp::TimeStep<uint_t>>, public with_dynamics_mixin
{

public:

    typedef DiscreteWorldBase<time_step_t>::action_type action_type;
    typedef DiscreteWorldBase<time_step_t>::time_step_type time_step_type;

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
     DiscreteWorldBase<time_step_t>("FrozenLake"),
     env_impl_("v0", gym_namespace)
{}


FrozenLakeEnv::time_step_type
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
    IterativePolicyEval<time_step_t> policy_eval(100, 1.0e-8, 1.0, env, policy);
    policy_eval.train();

    // save the value function into a csv file
    policy_eval.save("iterative_policy_eval.csv");

   return 0;
}
#else
#include <iostream>

int main() {

    std::cout<<"This example requires  gymfcpp. Configure cubeai to use gymfcpp"<<std::endl;
    return 0;
}
#endif


