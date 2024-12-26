#include "cubeai/base/cubeai_config.h"

#ifdef USE_RLENVS_CPP

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/mc/first_visit_mc.h"
#include "cubeai/rl/learning_rate_scheduler.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"
#include "cubeai/maths/vector_math.h"

#include "rlenvs/rlenvs_types_v2.h"
#include "rlenvs/envs/gymnasium/toy_text/frozen_lake_env.h"
#include "rlenvs/time_step.h"
#include "rlenvs/envs/envs_utils.h"

#include <iostream>
#include <random>


namespace rl_example_19
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::ConstantLRScheduler;
using cubeai::rl::policies::EpsilonGreedyPolicy;
using cubeai::rl::algos::mc::FirstVisitMCSolver;
using cubeai::rl::algos::mc::FirstVisitMCSolverConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using rlenvscpp::envs::gymnasium::FrozenLake;
using rlenvscpp::envs::gymnasium::FrozenLakeActionsEnum;

const std::string SERVER_URL = "http://0.0.0.0:8001/api";
const uint_t SEED = 42;


struct RandomActionSelector
{
    template<typename StateType>
    FrozenLake<4>::action_type operator()(const StateType state)const;
};

template<typename StateType>
FrozenLake<4>::action_type
RandomActionSelector::operator()(const StateType /*state*/)const{

    // randomly select an action
        std::mt19937 generator(SEED);
        std::uniform_int_distribution<> real_dist(0, 4);
        auto action = real_dist(generator);

        switch(action){

            case 0:
                return FrozenLakeActionsEnum::LEFT;
            case 1:
                return FrozenLakeActionsEnum::DOWN;
            case 2:
                return FrozenLakeActionsEnum::RIGHT;
            case 3:
                return FrozenLakeActionsEnum::UP;

            default:
                return FrozenLakeActionsEnum::INVALID_ACTION;
        }
}

struct TrajectoryGenerator
{
    std::vector<FrozenLake<4>::time_step_type>
    operator()(FrozenLake<4>&, uint_t max_steps);
};


std::vector<FrozenLake<4>::time_step_type>
TrajectoryGenerator::operator()(FrozenLake<4>& env, uint_t max_steps){

    RandomActionSelector action_selector;
    return rlenvscpp::envs::create_trajectory(env, action_selector, max_steps );
}


struct DiscountGenerator
{

    template<typename TrajectoryType>
    std::vector<real_t>
    operator()(const TrajectoryType&, uint_t max_steps);
};

template<typename TrajectoryType>
std::vector<real_t>
DiscountGenerator::operator()(const TrajectoryType&, uint_t max_steps)
{
    return cubeai::maths::logspace(0.0, static_cast<real_t>(max_steps), max_steps, 1.0);
}


typedef FrozenLake<4> env_type;
typedef TrajectoryGenerator trajectory_generator_type;
typedef ConstantLRScheduler learning_rate_scheduler_type;
typedef FirstVisitMCSolverConfig solver_config_type;
typedef DiscountGenerator discount_generator_type;




typedef FirstVisitMCSolver<env_type, trajectory_generator_type,
                           learning_rate_scheduler_type, discount_generator_type> solver_type;



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

    solver_config_type config;
    config.max_steps = 200;
    learning_rate_scheduler_type lr_scheduler;
    trajectory_generator_type trajectory_generator;
    discount_generator_type discount_generator;
    solver_type solver(config, trajectory_generator,lr_scheduler, discount_generator);

    RLSerialTrainerConfig trainer_config = {10, 10000, 1.0e-8};
    RLSerialAgentTrainer<env_type, solver_type> trainer(trainer_config, solver);

    auto info = trainer.train(env);
    std::cout<<"Trainer info: "<<std::endl;
    std::cout<<"\t"<<info<<std::endl;

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


