#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/td/q_learning.h"
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
using cubeai::rl::algos::td::QLearning;
using cubeai::rl::policies::EpsilonDecayOption;

typedef gymfcpp::TimeStep<uint_t> time_step_t;

class CliffWalkingEnv: public DiscreteWorldBase<time_step_t>
{
public:

    typedef  DiscreteWorldBase<time_step_t>::time_step_t time_step_t;
    typedef  DiscreteWorldBase<time_step_t>::action_t action_t;

    CliffWalkingEnv(gymfcpp::obj_t gym_namespace);
    ~CliffWalkingEnv()=default;

    virtual uint_t n_actions()const override final {return env_impl_.n_actions();}
    virtual uint_t n_copies()const override final{return 1;}
    virtual uint_t n_states()const override final {return env_impl_.n_states();}

    virtual std::vector<std::tuple<real_t, uint_t, real_t, bool>> transition_dynamics(uint_t s, uint_t aidx)const override final;
    virtual time_step_t on_episode(const action_t&)override final;
    virtual time_step_t reset() override final;
    virtual  void build(bool reset) override final;

private:

    gymfcpp::CliffWorld env_impl_;

};

CliffWalkingEnv::CliffWalkingEnv(gymfcpp::obj_t gym_namespace)
    :
      DiscreteWorldBase<time_step_t>("Cliffwalking"),
      env_impl_("v0", gym_namespace, false)
{}

std::vector<std::tuple<real_t, uint_t, real_t, bool>>
CliffWalkingEnv::transition_dynamics(uint_t s, uint_t aidx)const{
    return std::vector<std::tuple<real_t, uint_t, real_t, bool>>();
}

void
CliffWalkingEnv::build(bool reset_){
    env_impl_.make();

    if(reset_){
        reset();
    }
}

CliffWalkingEnv::time_step_t
CliffWalkingEnv::on_episode(const action_t& action){
    return env_impl_.on_episode(action);
}

CliffWalkingEnv::time_step_t
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

        EpsilonGreedyPolicy policy(1.0, env.n_actions(), EpsilonDecayOption::INVERSE_STEP);
        QLearning<time_step_t, EpsilonGreedyPolicy> qlearn(5000, 1.0e-8,
                                                            1.0, 0.01, 100, env, 1000, policy);

        qlearn.do_verbose_output();

        std::cout<<"Starting training..."<<std::endl;
        auto train_result = qlearn.train();

        std::cout<<train_result<<std::endl;
        std::cout<<"Finished training..."<<std::endl;

        std::cout<<"Saving value function..."<<std::endl;
        std::cout<<"Value function..."<<qlearn.value_func()<<std::endl;
        qlearn.save("qlearn_value_func.csv");
        qlearn.save_avg_scores("qlearn_avg_scores.csv");
        qlearn.save_state_action_function("qlearn_state_action_function.csv");

    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }
    return 0;
}
