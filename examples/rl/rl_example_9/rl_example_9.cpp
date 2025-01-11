#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/td/sarsa.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "rlenvs/envs/api_server/apiserver.h"
#include "rlenvs/envs/gymnasium/toy_text/cliff_world_env.h"

#include <boost/log/trivial.hpp>
#include <iostream>

namespace rl_example_9{

const std::string SERVER_URL = "http://0.0.0.0:8001/api";
const std::string SOLUTION_FILE = "sarsa_cliff_walking_v1.csv";

using cuberl::real_t;
using cuberl::uint_t;
using cuberl::rl::policies::EpsilonGreedyPolicy;
using cuberl::rl::algos::td::SarsaSolver;
using cuberl::rl::algos::td::SarsaConfig;
using cuberl::rl::policies::EpsilonDecayOption;
using cuberl::rl::RLSerialAgentTrainer;
using cuberl::rl::RLSerialTrainerConfig;
using rlenvscpp::envs::RESTApiServerWrapper;
typedef  rlenvscpp::envs::gymnasium::CliffWorld env_type;


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
    return action;
}

}

int main(){

	BOOST_LOG_TRIVIAL(info)<<"Starting agent training";
    using namespace rl_example_9;

    try{

		RESTApiServerWrapper server(SERVER_URL, true);
		
        // create the environment
        env_type env(server);

        BOOST_LOG_TRIVIAL(info)<<"Creating environment...";
        std::unordered_map<std::string, std::any> options;
        env.make("v0", options);
        env.reset();
        BOOST_LOG_TRIVIAL(info)<<"Done...";

        BOOST_LOG_TRIVIAL(info)<<"Number of states="<<env.n_states();
        BOOST_LOG_TRIVIAL(info)<<"Number of actions="<<env.n_actions();

        ActionSelector policy(1.0, env.n_actions()); //, EpsilonDecayOption::INVERSE_STEP);

        SarsaConfig sarsa_config;
        sarsa_config.gamma = 1.0;
        sarsa_config.eta = 0.01;
        sarsa_config.tolerance = 1.0e-8;
        sarsa_config.max_num_iterations_per_episode = 1000;
        sarsa_config.path = SOLUTION_FILE;

        SarsaSolver<env_type, ActionSelector> algorithm(sarsa_config, policy);

        RLSerialTrainerConfig trainer_config = {10, 500, 1.0e-8};

        RLSerialAgentTrainer<env_type,
                             SarsaSolver<env_type,
                             ActionSelector>> trainer(trainer_config, algorithm);

        auto info = trainer.train(env);
        BOOST_LOG_TRIVIAL(info)<<"Training info..."<<info;
		BOOST_LOG_TRIVIAL(info)<<"Finished agent training";

    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }

    return 0;
}
