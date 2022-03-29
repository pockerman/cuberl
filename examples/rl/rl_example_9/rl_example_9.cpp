#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/td/sarsa.h"
//#include "cubeai/rl/worlds/discrete_world.h"
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
using cubeai::rl::envs::DiscreteWorldBase;
using cubeai::rl::policies::EpsilonGreedyPolicy;
using cubeai::rl::algos::td::Sarsa;
using cubeai::rl::algos::td::SarsaConfig;
using cubeai::rl::policies::EpsilonDecayOption;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;

typedef gymfcpp::TimeStep<uint_t> time_step_type;

/*class CliffWalkingEnv: public DiscreteWorldBase<time_step_type>
{
public:

    typedef  DiscreteWorldBase<time_step_type>::time_step_type time_step_type;
    typedef  DiscreteWorldBase<time_step_type>::action_type action_type;

    CliffWalkingEnv(gymfcpp::obj_t gym_namespace);
    ~CliffWalkingEnv()=default;

    virtual uint_t n_actions()const override final {return env_impl_.n_actions();}
    virtual uint_t n_copies()const override final{return 1;}
    virtual uint_t n_states()const override final {return env_impl_.n_states();}

    virtual time_step_type step(const action_type&)override final;
    virtual time_step_type reset() override final;
    virtual  void build(bool reset) override final;

private:

    gymfcpp::CliffWorld env_impl_;

};

CliffWalkingEnv::CliffWalkingEnv(gymfcpp::obj_t gym_namespace)
    :
      DiscreteWorldBase<time_step_type>("Cliffwalking"),
      env_impl_("v0", gym_namespace, false)
{}

void
CliffWalkingEnv::build(bool reset_){
    env_impl_.make();

    if(reset_){
        reset();
    }
}

CliffWalkingEnv::time_step_type
CliffWalkingEnv::step(const action_type& action){
    return env_impl_.step(action);
}

CliffWalkingEnv::time_step_type
CliffWalkingEnv::reset(){
    return env_impl_.reset();
}
*/

}

int main(){

    using namespace example;

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
        Sarsa<gymfcpp::CliffWorld, EpsilonGreedyPolicy> sarsa(sarsa_config, policy);


        RLSerialTrainerConfig trainer_config = {10, 10000, 1.0e-8};

        RLSerialAgentTrainer<gymfcpp::CliffWorld,
                Sarsa<gymfcpp::CliffWorld, EpsilonGreedyPolicy>> trainer(trainer_config, algorithm);



        auto info = trainer.train(env);
        std::cout<<info<<std::endl;

        // save the value function into a csv file
        //algorithm.save("value_iteration_frozen_lake_v0.csv");

    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }

    return 0;
}
