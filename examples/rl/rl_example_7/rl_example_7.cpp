#include "cubeai/base/cubeai_config.h"

#ifdef USE_RLENVS_CPP

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/policy_iteration.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/policies/uniform_discrete_policy.h"
#include "cubeai/rl/policies/stochastic_adaptor_policy.h"

#include "rlenvs/envs/gymnasium/toy_text/frozen_lake_env.h"



#include <iostream>


namespace rl_example_7
{


const std::string SERVER_URL = "http://0.0.0.0:8001/api";

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::policies::UniformDiscretePolicy;
using cubeai::rl::policies::StochasticAdaptorPolicy;
using cubeai::rl::algos::dp::PolicyIterationSolver;
using cubeai::rl::algos::dp::PolicyIterationConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;

//typedef gymfcpp::TimeStep<uint_t> time_step_type;
using rlenvs_cpp::envs::gymnasium::FrozenLake;
typedef FrozenLake<4> env_type;
}

int main() {

    using namespace rl_example_7;

     // create the environment
    FrozenLake<4> env(SERVER_URL);
    std::unordered_map<std::string, std::any> options;

    std::cout<<"Creating the environment..."<<std::endl;
    env.make("v1", options);
    env.reset();
    std::cout<<"Done..."<<std::endl;

    UniformDiscretePolicy policy(env.n_states(), env.n_actions());
    StochasticAdaptorPolicy<UniformDiscretePolicy> policy_adaptor(env.n_states(), env.n_actions(), policy);

    PolicyIterationConfig config;
    config.gamma = 1.0;
    config.n_policy_eval_steps = 100;
    config.tolerance = 1.0e-8;

    typedef PolicyIterationSolver<env_type,
                                  UniformDiscretePolicy,
                                  StochasticAdaptorPolicy<UniformDiscretePolicy>> solver_type;

    solver_type solver(config, policy, policy_adaptor);
    /*PolicyIterationSolver<env_type, UniformDiscretePolicy,
                          StochasticAdaptorPolicy<UniformDiscretePolicy>> algorithm(config,
                                                                      policy, policy_adaptor);*/


    RLSerialTrainerConfig trainer_config = {10, 10000, 1.0e-8};

    RLSerialAgentTrainer<env_type, solver_type> trainer(trainer_config, solver);

    auto info = trainer.train(env);
    //std::cout<<info<<std::endl;

    // save the value function into a csv file
    solver.save("policy_iteration_frozen_lake_v0.csv");

    return 0;
}
#else
#include <iostream>

int main() {

    std::cout<<"This example requires  gymfcpp. Configure cubeai to use gymfcpp"<<std::endl;
    return 0;
}
#endif

