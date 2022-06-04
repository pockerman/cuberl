#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/policy_iteration.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/policies/uniform_discrete_policy.h"
#include "cubeai/rl/policies/stochastic_adaptor_policy.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/frozen_lake_env.h"
#include "gymfcpp/time_step.h"

#include <boost/python.hpp>

#include <iostream>


namespace rl_example_7
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::policies::UniformDiscretePolicy;
using cubeai::rl::policies::StochasticAdaptorPolicy;
using cubeai::rl::algos::dp::PolicyIteration;
using cubeai::rl::algos::dp::PolicyIterationConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;

//typedef gymfcpp::TimeStep<uint_t> time_step_type;
typedef rlenvs_cpp::gymfcpp::FrozenLake<4> env_type;

}

int main() {

    using namespace rl_example_7;

    Py_Initialize();
    auto gym_module = boost::python::import("__main__");
    auto gym_namespace = gym_module.attr("__dict__");

    env_type env("v0", gym_namespace);
    env.make();

    UniformDiscretePolicy policy(env.n_states(), env.n_actions());
    StochasticAdaptorPolicy<UniformDiscretePolicy> policy_adaptor(env.n_states(), env.n_actions(), policy);

    PolicyIterationConfig config;
    config.gamma = 1.0;
    config.n_policy_eval_steps = 100;
    config.tolerance = 1.0e-8;

    PolicyIteration<env_type, UniformDiscretePolicy,
            StochasticAdaptorPolicy<UniformDiscretePolicy>> algorithm(config,
                                                                      policy, policy_adaptor);


    RLSerialTrainerConfig trainer_config = {10, 10000, 1.0e-8};

    RLSerialAgentTrainer<env_type,
            PolicyIteration<env_type,
                            UniformDiscretePolicy,
                            StochasticAdaptorPolicy<UniformDiscretePolicy>>> trainer(trainer_config, algorithm);

    auto info = trainer.train(env);
    std::cout<<info<<std::endl;

    // save the value function into a csv file
    algorithm.save("policy_iteration_frozen_lake_v0.csv");

    return 0;
}


