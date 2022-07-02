/**
  * Double Q-learning on CartPole-v0 environment
  *
 */


#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/math_constants.h"
#include "cubeai/rl/policies/epsilon_double_qtable_greedy_policy.h"
#include "cubeai/rl/epsilon_decay_options.h"
#include "cubeai/rl/episode_info.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/rl_mixins.h"
#include "gymfcpp/state_aggregation_cart_pole_env.h"

#include <boost/python.hpp>

#include <vector>
#include <tuple>
#include <map>
#include <iostream>
#include <cmath>
#include <exception>

#include <chrono>
#include <random>


namespace rl_example_18{


using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynVec;
using cubeai::DynMat;
using cubeai::rl::policies::EpsilonDoubleQTableGreedyPolicy;
using cubeai::rl::EpsilonDecayOptionType;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using cubeai::rl::EpisodeInfo;
using rlenvs_cpp::gymfcpp::StateAggregationCartPole;

const real_t EPS = 0.1;
const real_t GAMMA = 1.0;
const real_t ALPHA = 0.1;



struct DoubleQLearningConfig
{

    std::string path{""};
    real_t tolerance;
    real_t gamma;
    real_t eta;
    uint_t max_num_iterations_per_episode;
    uint_t n_episodes;
    uint_t seed{42};

};

typedef StateAggregationCartPole::state_type state_type;
typedef std::map<state_type, DynVec<real_t>> table_type;

template<typename ActionSelector>
class DoubleQLearning final: protected cubeai::rl::with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>,
                             protected cubeai::rl::with_double_q_table_max_action_mixin
{
public:

    typedef StateAggregationCartPole env_type;
    typedef uint_t action_type;
    typedef std::tuple<uint_t, uint_t, uint_t, uint_t> state_type;
    typedef ActionSelector action_selector_type;


    DoubleQLearning(const DoubleQLearningConfig config, const ActionSelector& selector);


    void actions_before_training_begins(env_type&);
    void actions_after_training_ends(env_type&){}
    void actions_before_episode_begins(env_type&, uint_t /*episode_idx*/){}
    void actions_after_episode_ends(env_type&, uint_t episode_idx, const cubeai::rl::EpisodeInfo&){ action_selector_.adjust_on_episode(episode_idx);}

    cubeai::rl::EpisodeInfo on_training_episode(env_type&, uint_t episode_idx);

    void save(std::string filename)const{}

private:

    DoubleQLearningConfig config_;
    action_selector_type action_selector_;
    void update_q_table_(const action_type& action, const state_type& cstate,
                         const state_type& next_state, real_t reward, env_type& env);

};

template<typename ActionSelector>
DoubleQLearning<ActionSelector>::DoubleQLearning(const DoubleQLearningConfig config, const ActionSelector& selector)
    :
     cubeai::rl::with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>(),
     config_(config),
     action_selector_(selector)
{}


template<typename ActionSelector>
void
DoubleQLearning<ActionSelector>::actions_before_training_begins(env_type& env){
    this->with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>::initialize(env.get_states(), env.n_actions(), 0.0);
}

template<typename ActionSelector>
cubeai::rl::EpisodeInfo
DoubleQLearning<ActionSelector>::on_training_episode(env_type& env, uint_t episode_idx){

    auto start = std::chrono::steady_clock::now();
    EpisodeInfo info;

    // total score for the episode
    auto episode_score = 0.0;

    auto state = env.reset().observation();

    uint_t itr=0;
    for(;  itr < config_.max_num_iterations_per_episode; ++itr){

        // select an action
        auto action = action_selector_(this->with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>::q_table_1,
                                       this->with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>::q_table_2, state);

        // Take an action on the environment
        auto step_type_result = env.step(action);

        auto next_state = step_type_result.observation();
        auto reward = step_type_result.reward();
        auto done = step_type_result.done();

        // accumulate score
        episode_score += reward;

        // update the table
        update_q_table_(action, state, next_state, reward, env);
        state = next_state;

        if(done){
            break;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<real_t> elapsed_seconds = end-start;

    info.episode_index = episode_idx;
    info.episode_reward = episode_score;
    info.episode_iterations = itr;
    info.total_time = elapsed_seconds;
    return info;

}

template <typename ActionSelector>
void
DoubleQLearning<ActionSelector>::update_q_table_(const action_type& action, const state_type& cstate,
                                                 const state_type& next_state, real_t reward, env_type& env){

    // flip a coin 50% of the time we update Q1
    // whilst 50% of the time Q2
    std::mt19937 gen(config_.seed); //rd());

    // generate a number in [0, 1]
    std::uniform_real_distribution<> real_dist_(0.0, 1.0);

    // update Q1
    if(real_dist_(gen) <= 0.5){

        // the current qvalue
        auto q_current = this->with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>::template get<1>(cstate, action);
        auto Qsa_next = 0.0;

        //if(this->env_ref_().is_valid_state(next_state)){
            auto max_act = this->with_double_q_table_max_action_mixin::max_action(this->with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>::q_table_1,
                                                                                  next_state, env.n_actions());

            // value of next state
            Qsa_next = this->with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>::template get<2>(next_state, max_act);
         //}

         // construct TD target
         auto target = reward + (config_.gamma * Qsa_next);

         // get updated value
         auto new_value = q_current + (config_.eta * (target - q_current));
         this->with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>::template set<1>(cstate, action, new_value);
    }
    else{

        // the current qvalue
        auto q_current = this->with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>::template get<2>(cstate, action);
        auto Qsa_next = 0.0;


        auto max_act = this->with_double_q_table_max_action_mixin::max_action(this->with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>::q_table_2,
                                                                                  next_state, env.n_actions());

            // value of next state
        Qsa_next = this->with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>::template get<1>(next_state, max_act);


         // construct TD target
         auto target = reward + (config_.gamma * Qsa_next);

         // get updated value
         auto new_value = q_current + (config_.eta * (target - q_current));
         this->with_double_q_table_mixin<std::map<state_type, DynVec<real_t>>>::template set<2>(cstate, action, new_value);
    }
}



}


int main(){

    using namespace rl_example_18;

    try{

        Py_Initialize();
        auto main_module = boost::python::import("__main__");
        auto main_namespace = main_module.attr("__dict__");

        // create the environment
        StateAggregationCartPole env("v0", main_namespace, 10);

        // the policy to use
        EpsilonDoubleQTableGreedyPolicy<table_type> policy(EPS, env.n_actions(), EpsilonDecayOptionType::NONE);

        // configuration for the algorithm
        DoubleQLearningConfig config;
        config.eta = ALPHA;
        config.gamma = GAMMA;
        config.n_episodes = 50000;
        config.max_num_iterations_per_episode = 10000;

        // the agent to traain
        DoubleQLearning<EpsilonDoubleQTableGreedyPolicy<table_type>> algorithm(config, policy);

        RLSerialTrainerConfig trainer_config = {100, 50000, 1.0e-8};

        RLSerialAgentTrainer<StateAggregationCartPole,
                             DoubleQLearning<EpsilonDoubleQTableGreedyPolicy<table_type>>> trainer(trainer_config, algorithm);

        auto info = trainer.train(env);
        std::cout<<info<<std::endl;


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
