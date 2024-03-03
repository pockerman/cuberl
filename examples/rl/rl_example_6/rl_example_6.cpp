#include "cubeai/base/cubeai_config.h"

#ifdef USE_RLENVS_CPP

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/iterative_policy_evaluation.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/policies/uniform_discrete_policy.h"

#include "rlenvs/rlenvs_types_v2.h"
#include "rlenvs/envs/gymnasium/toy_text/frozen_lake_env.h"
#include "rlenvs/time_step.h"

#include <iostream>

namespace rl_example_6
{

const std::string SERVER_URL = "http://0.0.0.0:8001/api";

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::policies::UniformDiscretePolicy;
using cubeai::rl::algos::dp::IterativePolicyEval;
using cubeai::rl::algos::dp::IterativePolicyEvalConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using rlenvs_cpp::envs::gymnasium::FrozenLake;
typedef FrozenLake<4> env_type;

}

int main() {

    using namespace rl_example_6;


    // create the environment
    FrozenLake<4> env(SERVER_URL);

    std::cout<<"Environment URL: "<<env.get_url()<<std::endl;

    std::unordered_map<std::string, std::any> options;
    env.make("v0", options);
    env.reset();


    /*UniformDiscretePolicy policy(env.n_states(), env.n_actions());
    IterativePolicyEvalConfig config;
    config.tolerance = 1.0e-8;

    IterativePolicyEval<env_type, UniformDiscretePolicy> algorithm(config, policy);

    RLSerialTrainerConfig trainer_config = {10, 10000, 1.0e-8};

    RLSerialAgentTrainer<env_type, IterativePolicyEval<env_type,
            UniformDiscretePolicy>> trainer(trainer_config, algorithm);

    auto info = trainer.train(env);
    std::cout<<info<<std::endl;

    // save the value function for plotting
    algorithm.save("iterative_policy_evaluation_frozen_lake.csv");*/

   return 0;
}
#else
#include <iostream>

int main() {

    std::cout<<"This example requires  gymfcpp. Configure cubeai to use gymfcpp"<<std::endl;
    return 0;
}
#endif


