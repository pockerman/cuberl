#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/td/q_learning.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/cliff_world_env.h"
#include "gymfcpp/time_step.h"

#include <deque>
#include <iostream>


namespace example{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::policies::EpsilonGreedyPolicy;
using cubeai::rl::algos::td::QLearning;
using cubeai::rl::algos::td::QLearningConfig;
using cubeai::rl::policies::EpsilonDecayOption;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;

typedef rlenvs_cpp::gymfcpp::CliffWorld env_type;
}


int main(){

    using namespace example;

    try{

        Py_Initialize();
        auto gym_module = boost::python::import("__main__");
        auto gym_namespace = gym_module.attr("__dict__");

        env_type env("v0", gym_namespace);
        env.make();

        std::cout<<"Number of states="<<env.n_states()<<std::endl;
        std::cout<<"Number of actions="<<env.n_actions()<<std::endl;

        EpsilonGreedyPolicy policy(1.0, env.n_actions(), EpsilonDecayOption::INVERSE_STEP);

        QLearningConfig qlearn_config;
        qlearn_config.gamma = 1.0;
        qlearn_config.eta = 0.01;
        qlearn_config.tolerance = 1.0e-8;
        qlearn_config.max_num_iterations_per_episode = 1000;
        qlearn_config.path = "qlearning_cliff_walking_v0.csv";

        QLearning<env_type, EpsilonGreedyPolicy> algorithm(qlearn_config, policy);

        RLSerialTrainerConfig trainer_config = {10, 10000, 1.0e-8};

        RLSerialAgentTrainer<env_type,
                QLearning<env_type, EpsilonGreedyPolicy>> trainer(trainer_config, algorithm);

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
