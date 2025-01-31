#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/policy_iteration.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/policies/uniform_discrete_policy.h"
#include "rlenvs/envs/api_server/apiserver.h"
#include "rlenvs/envs/gymnasium/toy_text/frozen_lake_env.h"


#include <boost/log/trivial.hpp>
#include <iostream>


namespace rl_example_7
{


const std::string SERVER_URL = "http://0.0.0.0:8001/api";
const std::string SOLUTION_FILE = "policy_iteration_frozen_lake_v0.csv";

using cuberl::real_t;
using cuberl::uint_t;
using cuberl::rl::policies::UniformDiscretePolicy;
using cuberl::rl::algos::dp::PolicyIterationSolver;
using cuberl::rl::algos::dp::PolicyIterationConfig;
using cuberl::rl::RLSerialAgentTrainer;
using cuberl::rl::RLSerialTrainerConfig;
using rlenvscpp::envs::gymnasium::FrozenLake;
using rlenvscpp::envs::RESTApiServerWrapper;

typedef FrozenLake<4> env_type;
}

int main() {

	BOOST_LOG_TRIVIAL(info)<<"Starting agent training";
    using namespace rl_example_7;
	
	RESTApiServerWrapper server(SERVER_URL, true);

     // create the environment
    FrozenLake<4> env(server);
    std::unordered_map<std::string, std::any> options;

	BOOST_LOG_TRIVIAL(info)<<"Creating environment...";
	
    env.make("v1", options);
    env.reset();
    BOOST_LOG_TRIVIAL(info)<<"Done...";

    UniformDiscretePolicy policy(env.n_states(), env.n_actions());
    
    PolicyIterationConfig config;
    config.gamma = 1.0;
    config.n_policy_eval_steps = 15;
    config.tolerance = 1.0e-8;

    typedef PolicyIterationSolver<env_type,
                                  UniformDiscretePolicy> solver_type;

    solver_type solver(config, env.n_actions(), policy);
    
	// output message frequence, number of episodes, tolerance
    RLSerialTrainerConfig trainer_config = {10, 100, 1.0e-8};

    RLSerialAgentTrainer<env_type, solver_type> trainer(trainer_config, solver);

    auto info = trainer.train(env);
    
	BOOST_LOG_TRIVIAL(info)<<"Saving solution to "<<SOLUTION_FILE;
	
    // save the value function into a csv file
    solver.save(SOLUTION_FILE);

	BOOST_LOG_TRIVIAL(info)<<"Finished agent training";
    return 0;
}
