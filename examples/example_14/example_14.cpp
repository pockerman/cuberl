#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/td/expected_sarsa.h"
#include "cubeai/rl/worlds/discrete_world.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/cliff_world.h"
#include "gymfcpp/time_step.h"

#include <deque>
#include <iostream>


namespace example{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::envs::DiscreteWorldBase;
using cubeai::rl::policies::EpsilonGreedyPolicy;
using cubeai::rl::algos::td::ExpectedSARSA;
using cubeai::rl::policies::EpsilonDecayOption;

typedef gymfcpp::TimeStep<uint_t> time_step_type;

class CliffWalkingEnv: public DiscreteWorldBase<time_step_type>
{
public:

    typedef  DiscreteWorldBase<time_step_type>::time_step_type time_step_type;
    typedef  DiscreteWorldBase<time_step_type>::action_type action_type;

    CliffWalkingEnv(gymfcpp::obj_t gym_namespace);
    ~CliffWalkingEnv()=default;

    virtual uint_t n_actions()const override final {return env_impl_.n_actions();}
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

}


int main(){

    using namespace example;

    try{

        Py_Initialize();
        auto gym_module = boost::python::import("gym");
        auto gym_namespace = gym_module.attr("__dict__");

        CliffWalkingEnv env(gym_namespace);
        env.build(true);

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
