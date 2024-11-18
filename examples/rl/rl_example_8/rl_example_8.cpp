#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/value_iteration.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/policies/uniform_discrete_policy.h"
#include "cubeai/rl/policies/stochastic_adaptor_policy.h"

#include "rlenvs/envs/gymnasium/toy_text/frozen_lake_env.h"

#include <cmath>
#include <utility>
#include <tuple>
#include <iostream>
#include <random>
#include <algorithm>

namespace rl_example_8
{

const std::string SERVER_URL = "http://0.0.0.0:8001/api";
const std::string SOLUTION_FILE = "value_iteration_frozen_lake_v1.csv";

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::policies::UniformDiscretePolicy;
using cubeai::rl::policies::StochasticAdaptorPolicy;
using cubeai::rl::algos::dp::ValueIteration;
using cubeai::rl::algos::dp::ValueIterationConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;

using rlenvs_cpp::envs::gymnasium::FrozenLake;
typedef FrozenLake<4> env_type;

typedef  ValueIteration<env_type,
                        UniformDiscretePolicy,
                        StochasticAdaptorPolicy<UniformDiscretePolicy>> solver_type;

}

int main() {

	BOOST_LOG_TRIVIAL(info)<<"Starting agent training";
	
    using namespace rl_example_8;

    // create the environment
    env_type env(SERVER_URL);

    BOOST_LOG_TRIVIAL(info)<<"Creating environment...";
    std::unordered_map<std::string, std::any> options;

    env.make("v1", options);
    env.reset();
    BOOST_LOG_TRIVIAL(info)<<"Done...";

    // start with a uniform random policy i.e.
    // the agnet knows nothing about the environment
    UniformDiscretePolicy policy(env.n_states(), env.n_actions());

    StochasticAdaptorPolicy<UniformDiscretePolicy> policy_adaptor(env.n_states(), env.n_actions(), policy);

    ValueIterationConfig config;
    config.gamma = 1.0;
    config.tolerance = 1.0e-8;

    solver_type algorithm(config, policy, policy_adaptor);

    RLSerialTrainerConfig trainer_config = {10, 10000, 1.0e-8};

    RLSerialAgentTrainer<env_type, solver_type> trainer(trainer_config, algorithm);

    auto info = trainer.train(env);
    std::cout<<info<<std::endl;

	BOOST_LOG_TRIVIAL(info)<<"Saving solution to "<<SOLUTION_FILE;

    // save the value function into a csv file
    algorithm.save(SOLUTION_FILE);

	BOOST_LOG_TRIVIAL(info)<<"Finished agent training";
    return 0;
}


