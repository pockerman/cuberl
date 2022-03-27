#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/value_iteration.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
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

namespace rl_example_8
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::policies::UniformDiscretePolicy;
using cubeai::rl::policies::StochasticAdaptorPolicy;
using cubeai::rl::algos::dp::ValueIteration;
using cubeai::rl::algos::dp::ValueIterationConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;


}

int main() {

    using namespace rl_example_8;

    Py_Initialize();
    auto gym_module = boost::python::import("__main__");
    auto gym_namespace = gym_module.attr("__dict__");

    gymfcpp::FrozenLake<4> env("v0", gym_namespace);
    env.make();

    // start with a uniform random policy i.e.
    // the agnet knows nothing about the environment
    UniformDiscretePolicy policy(env.n_states(), env.n_actions());

    StochasticAdaptorPolicy<UniformDiscretePolicy> policy_adaptor(env.n_states(), env.n_actions(), policy);

    ValueIterationConfig config;
    config.gamma = 1.0;
    config.tolerance = 1.0e-8;

    ValueIteration<gymfcpp::FrozenLake<4>, UniformDiscretePolicy,
            StochasticAdaptorPolicy<UniformDiscretePolicy> > algorithm(config, policy, policy_adaptor);

    RLSerialTrainerConfig trainer_config = {10, 10000, 1.0e-8};

    RLSerialAgentTrainer<gymfcpp::FrozenLake<4>,
            ValueIteration<gymfcpp::FrozenLake<4>,
                            UniformDiscretePolicy,
                            StochasticAdaptorPolicy<UniformDiscretePolicy>>> trainer(trainer_config, algorithm);

    auto info = trainer.train(env);
    std::cout<<info<<std::endl;

    // save the value function into a csv file
    algorithm.save("value_iteration_frozen_lake_v0.csv");

    return 0;
}


