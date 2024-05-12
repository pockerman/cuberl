#include "cubeai/base/cubeai_config.h"

#ifdef USE_RLENVS_CPP

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/td/expected_sarsa.h"
#include "cubeai/rl/worlds/discrete_world.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"

#include "rlenvs/envs/gymnasium/toy_text/cliff_world_env.h"

#include <iostream>
#include <unordered_map>


namespace rl_example_14{

const std::string SERVER_URL = "http://0.0.0.0:8001/api";

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::policies::EpsilonGreedyPolicy;
using cubeai::rl::algos::td::ExpectedSARSA;
using cubeai::rl::algos::td::QLearningConfig;
using cubeai::rl::policies::EpsilonDecayOption;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using rlenvs_cpp::envs::gymnasium::CliffWorldActionsEnum;
typedef  rlenvs_cpp::envs::gymnasium::CliffWorld env_type;

}


int main(){

    using namespace rl_example_14;

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

        EpsilonGreedyPolicy policy(0.005, env.n_actions(), EpsilonDecayOption::NONE);
        ExpectedSARSA<CliffWalkingEnv, EpsilonGreedyPolicy> expected_sarsa(5000, 1.0e-8,
                                                                             1.0, 0.01, 100, env, 1000, policy);

        expected_sarsa.do_verbose_output();

        std::cout<<"Starting training..."<<std::endl;
        auto train_result = expected_sarsa.train();

        std::cout<<train_result<<std::endl;
        std::cout<<"Finished training..."<<std::endl;

        std::cout<<"Saving value function..."<<std::endl;
        std::cout<<"Value function..."<<expected_sarsa.value_func()<<std::endl;

        expected_sarsa.save("expected_sarsa_value_func.csv");
        expected_sarsa.save_avg_scores("expected_sarsa_avg_scores.csv");
        expected_sarsa.save_state_action_function("expected_sarsa_state_action_function.csv");

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
