#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/td/sarsa.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/cliff_world_env.h"
#include "gymfcpp/time_step.h"

#include <iostream>

namespace rl_example_9{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::policies::EpsilonGreedyPolicy;
using cubeai::rl::algos::td::Sarsa;
using cubeai::rl::algos::td::SarsaConfig;
using cubeai::rl::policies::EpsilonDecayOption;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;

}

int main(){

    using namespace rl_example_9;

    try{

        Py_Initialize();
        auto gym_module = boost::python::import("__main__");
        auto gym_namespace = gym_module.attr("__dict__");

        gymfcpp::CliffWorld env("v0", gym_namespace);
        env.make();

        std::cout<<"Number of states="<<env.n_states()<<std::endl;
        std::cout<<"Number of actions="<<env.n_actions()<<std::endl;

        EpsilonGreedyPolicy policy(1.0, env.n_actions(), EpsilonDecayOption::INVERSE_STEP);

        SarsaConfig sarsa_config;
        sarsa_config.gamma = 1.0;
        sarsa_config.eta = 0.01;
        sarsa_config.tolerance = 1.0e-8;
        sarsa_config.max_num_iterations_per_episode = 1000;
        sarsa_config.path = "sarsa_cliff_walking_v0.csv";

        Sarsa<gymfcpp::CliffWorld, EpsilonGreedyPolicy> algorithm(sarsa_config, policy);

        RLSerialTrainerConfig trainer_config = {10, 10000, 1.0e-8};

        RLSerialAgentTrainer<gymfcpp::CliffWorld,
                Sarsa<gymfcpp::CliffWorld, EpsilonGreedyPolicy>> trainer(trainer_config, algorithm);

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
