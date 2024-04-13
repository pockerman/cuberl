#include "cubeai/base/cubeai_config.h"

#ifdef USE_RLENVS_CPP

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/mc/first_visit_mc.h"
#include "cubeai/rl/learning_rate_scheduler.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"

#include "rlenvs/rlenvs_types_v2.h"
#include "rlenvs/envs/gymnasium/toy_text/frozen_lake_env.h"
#include "rlenvs/time_step.h"

#include <iostream>

namespace rl_example_19
{

const std::string SERVER_URL = "http://0.0.0.0:8001/api";

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::ConstantLRScheduler;
using cubeai::rl::policies::EpsilonGreedyPolicy;
using cubeai::rl::algos::mc::FirstVisitMCSolver;
using cubeai::rl::algos::mc::FirstVisitMCSolverConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using rlenvs_cpp::envs::gymnasium::FrozenLake;

struct TrajectoryGenerator
{

    template<typename PolicyType>
    std::vector<FrozenLake<4>::time_step_type> operator()(FrozenLake<4>&, PolicyType&, uint_t)
    {return std::vector<FrozenLake<4>::time_step_type>();}

};



typedef FrozenLake<4> env_type;
typedef EpsilonGreedyPolicy policy_type;
typedef TrajectoryGenerator trajectory_generator_type;
typedef ConstantLRScheduler learning_rate_scheduler_type;
typedef FirstVisitMCSolverConfig solver_config_type;




typedef FirstVisitMCSolver<env_type, policy_type, trajectory_generator_type,
                           learning_rate_scheduler_type> solver_type;



}

int main() {

    using namespace rl_example_19;

    // create the environment
    FrozenLake<4> env(SERVER_URL);

    std::cout<<"Environment URL: "<<env.get_url()<<std::endl;
    std::unordered_map<std::string, std::any> options;

    std::cout<<"Creating the environment..."<<std::endl;
    env.make("v1", options);
    env.reset();
    std::cout<<"Done..."<<std::endl;

    EpsilonGreedyPolicy policy(0.1);

    solver_config_type config;
    learning_rate_scheduler_type lr_scheduler;
    trajectory_generator_type trajectory_generator;
    solver_type solver(config, policy, trajectory_generator,lr_scheduler);

    RLSerialTrainerConfig trainer_config = {10, 10000, 1.0e-8};

    RLSerialAgentTrainer<env_type,
                        solver_type> trainer(trainer_config, solver);

    auto info = trainer.train(env);
    std::cout<<info<<std::endl;

    // save the value function for plotting
    //algorithm.save("iterative_policy_evaluation_frozen_lake.csv");

   return 0;
}
#else
#include <iostream>

int main() {

    std::cout<<"This example requires  gymfcpp. Configure cubeai to use gymfcpp"<<std::endl;
    return 0;
}
#endif


