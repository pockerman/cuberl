#include "cubeai/base/cubeai_config.h"

#ifdef USE_RLENVS_CPP

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/td/sarsa.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"


#include "rlenvs/envs/gymnasium/toy_text/cliff_world_env.h"
#include <iostream>

namespace rl_example_9{

const std::string SERVER_URL = "http://0.0.0.0:8001/api";

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::policies::EpsilonGreedyPolicy;
using cubeai::rl::algos::td::SarsaSolver;
using cubeai::rl::algos::td::SarsaConfig;
using cubeai::rl::policies::EpsilonDecayOption;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using rlenvs_cpp::envs::gymnasium::CliffWorldActionsEnum;
typedef  rlenvs_cpp::envs::gymnasium::CliffWorld env_type;

// ActionSelector. This is a simple wrapper to
// EpsilonGreedyPolicy class in order to adapt the
// returned action to the appropriate Env::ENUM

struct ActionSelector
{

    ActionSelector(real_t eps, uint_t n_actions);

    template<typename MapType>
    env_type::action_type
    operator()(const MapType& q_map, uint_t state)const;

    // the underlying policy
    EpsilonGreedyPolicy policy_;

};

ActionSelector::ActionSelector(real_t eps, uint_t n_actions)
:
policy_(eps, n_actions,EpsilonDecayOption::INVERSE_STEP)
{}


template<typename MapType>
env_type::action_type
ActionSelector::operator()(const MapType& q_map, uint_t state)const{

    auto action = policy_(q_map, state);

    // convert to
    switch(action){

        case 0:
            return CliffWorldActionsEnum::UP;
        case 1:
            return CliffWorldActionsEnum::RIGHT;
        case 2:
            return CliffWorldActionsEnum::DOWN;
        case 3:
            return CliffWorldActionsEnum::LEFT;

    }

    return CliffWorldActionsEnum::INVALID_ACTION;
}

}

int main(){

    using namespace rl_example_9;

    try{

        // create the environment
        env_type env(SERVER_URL);

        std::cout<<"Environment URL: "<<env.get_url()<<std::endl;
        std::unordered_map<std::string, std::any> options;

        std::cout<<"Creating the environment..."<<std::endl;
        env.make("v1", options);
        env.reset();
        std::cout<<"Done..."<<std::endl;

        std::cout<<"Number of states="<<env.n_states()<<std::endl;
        std::cout<<"Number of actions="<<env.n_actions()<<std::endl;

        ActionSelector policy(1.0, env.n_actions()); //, EpsilonDecayOption::INVERSE_STEP);

        SarsaConfig sarsa_config;
        sarsa_config.gamma = 1.0;
        sarsa_config.eta = 0.01;
        sarsa_config.tolerance = 1.0e-8;
        sarsa_config.max_num_iterations_per_episode = 1000;
        sarsa_config.path = "sarsa_cliff_walking_v0.csv";

        SarsaSolver<env_type, ActionSelector> algorithm(sarsa_config, policy);

        RLSerialTrainerConfig trainer_config = {10, 10000, 1.0e-8};

        RLSerialAgentTrainer<env_type,
                             SarsaSolver<env_type,
                             ActionSelector>> trainer(trainer_config, algorithm);

        auto info = trainer.train(env);
        std::cout<<info<<std::endl;

    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }

    return 0;
}

#else
#include <iostream>

int main(){

    std::cout<<"This example requires the flag USE_RLENVS_CPP to be true."<<std::endl;
    std::cout<<"Reconfigures and rebuild the library by setting the flag USE_RLENVS_CPP  to ON."<<std::endl;
    return 1;
}
#endif
