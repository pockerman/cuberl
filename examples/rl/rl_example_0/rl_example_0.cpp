/**
  * Example 0: Demonstrates the use of the DummyAlgorithm class.
  * This class exposes the basic API that most implemented RL
  * algorithms expose.
  *
  * */


#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dummy/dummy_algorithm.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/algorithms/rl_algo_config.h"
#include "cubeai/rl/agents/dummy_agent.h"
#include "cubeai/utils/iteration_counter.h"
#include "cubeai/utils/cubeai_concepts.h"
#include "cubeai/extern/matplotlibcpp.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/mountain_car_env.h"
#include "gymfcpp/time_step.h"
#include "gymfcpp/render_mode_enum.h"

#include <iostream>


namespace example_0
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynMat;
using cubeai::DynVec;
using cubeai::rl::algos::DummyAlgorithm;
using cubeai::rl::algos::DummyAlgorithmConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using cubeai::rl::agents::DummyAgent;
using cubeai::utils::IterationCounter;
using rlenvs_cpp::gymfcpp::MountainCar;


template<cubeai::utils::concepts::float_or_integral_vector PolicyValuesType, typename StateType>
class DummyPolicy
{
public:

    typedef PolicyValuesType policy_values_type;
    typedef typename policy_values_type::value_type action_type;
    typedef StateType state_type;

    ///
    /// \brief DeterministicActionPolicy
    /// \param values
    ///
    explicit DummyPolicy(policy_values_type&& values);

    ///
    ///
    ///
    const action_type& on_state(const state_type& state)const;

private:

    policy_values_type policy_values_;
    mutable uint_t counter_;

};


template<cubeai::utils::concepts::float_or_integral_vector PolicyValuesType, typename StateType>
DummyPolicy<PolicyValuesType, StateType>::DummyPolicy(policy_values_type&& values)
    :
    policy_values_(values)
{}

template<cubeai::utils::concepts::float_or_integral_vector PolicyValuesType, typename StateType>
const typename DummyPolicy<PolicyValuesType, StateType>::action_type&
DummyPolicy<PolicyValuesType, StateType>::on_state(const state_type& /*state*/)const{

    if(counter_ >= policy_values_.size()){
        counter_ = 0;
    }

    return policy_values_[counter_++];
}


template<typename EnvType>
class Criteria
{

public:

    typedef EnvType env_type;

    explicit Criteria(env_type& env, uint_t n_max_itsr);

    bool continue_iterations()noexcept;

private:

    env_type& env_;
    IterationCounter counter_;

};

template<typename EnvType>
Criteria<EnvType>::Criteria(env_type& env, uint_t n_max_itsrs)
    :
      env_(env),
      counter_(n_max_itsrs)
{}

template<typename EnvType>
bool
Criteria<EnvType>::continue_iterations()noexcept{

    if(counter_.continue_iterations()){
        env_.render(rlenvs_cpp::RenderModeType::human);
        return true;
    }

    return false;
}

}

int main() {

    using namespace example_0;

    try{

        Py_Initialize();
        auto main_module = boost::python::import("__main__");
        auto gym_namespace = main_module.attr("__dict__");

        // create the environment
        MountainCar env("v0", gym_namespace, false);
        env.make();
        env.reset();


        // The episode terminates after 200
        // steps: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
        DummyAlgorithmConfig config = {200};
        DummyAlgorithm<MountainCar> algorithm(config);

        RLSerialTrainerConfig trainer_config = {100, 10000, 1.0e-8};

        RLSerialAgentTrainer<MountainCar, DummyAlgorithm<MountainCar>> trainer(trainer_config, algorithm);

        auto info = trainer.train(env);
        std::cout<<info<<std::endl;

        // plot the rewarda
        const auto& rewards = trainer.episodes_total_rewards();

        typedef DummyPolicy<DummyAlgorithm<MountainCar>::policy_type, MountainCar::state_type> policy_type;
        policy_type policy(std::move(algorithm.get_policy()));

        // let's play initialize and agent with the policy
        DummyAgent<MountainCar, policy_type> agent(policy);

        Criteria<MountainCar> criteria(env, 100000);
        env.reset();
        agent.play(env, criteria);

        namespace plt = matplotlibcpp;
        plt::plot(rewards, "r-");
        plt::xlabel("Episode");
        plt::ylabel("Reward");
        plt::title("Dummy algorithm episode reward");
        plt::show();
        plt::save("dummy_agent_reward.png");
    }
    catch(const boost::python::error_already_set&)
    {
            PyErr_Print();
    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }

   return 0;
}


