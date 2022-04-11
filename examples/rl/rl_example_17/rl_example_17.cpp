#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_consts.h"

#include "cubeai/rl/episode_info.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/epsilon_decay_options.h"

#include "gymfcpp/state_aggregation_cart_pole_env.h"
#include <boost/python.hpp>

#include <vector>
#include <iostream>


namespace rl_example_17{


using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynVec;
using cubeai::DynMat;

using cubeai::rl::EpsilonDecayOptionType;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using cubeai::rl::EpisodeInfo;
using gymfcpp::StateAggregationCartPole;

const real_t EPS = 0.1;
const real_t GAMMA = 1.0;
const real_t ALPHA = 0.1;

typedef StateAggregationCartPole::state_type state_type;
typedef std::map<state_type, DynVec<real_t>> table_type;


struct QLearningConfig
{

    std::string path{""};
    real_t tolerance;
    real_t gamma;
    real_t eta;
    uint_t max_num_iterations_per_episode;
    uint_t n_episodes;
    uint_t seed{42};

};


class EpsilonGreedyPolicy
{

public:

    EpsilonGreedyPolicy(real_t eps, uint_t n_actions, EpsilonDecayOptionType decay_type);

    void adjust_on_episode(uint_t episode_idx)noexcept;

    ///
    /// \brief operator()
    ///
    uint_t operator()(const table_type& q_map, const state_type& state)const;

private:

    real_t eps_init_;
    real_t eps_;
    real_t min_eps_;
    real_t max_eps_;
    real_t epsilon_decay_;
    uint_t n_actions_;
    uint_t seed_;
    EpsilonDecayOptionType decay_op_;


};

EpsilonGreedyPolicy::EpsilonGreedyPolicy(real_t eps, uint_t n_actions, EpsilonDecayOptionType decay_op)
    :
      eps_init_(eps),
      eps_(eps),
      min_eps_(0.001),
      max_eps_(1.0),
      epsilon_decay_(0.01),
      n_actions_(n_actions),
      seed_(42),
      decay_op_(decay_op)
{}

void
EpsilonGreedyPolicy::adjust_on_episode(uint_t episode)noexcept{


    switch(decay_op_)
    {
    case EpsilonDecayOptionType::NONE:
    {
        return;
    }
    case EpsilonDecayOptionType::INVERSE_STEP:
    {
        if(episode == 0){
            episode = 1;
        }

        // there are various methods to do epsilon
        // reduction
        eps_ = 1.0 / episode;

        if(eps_ < min_eps_){
            eps_ = min_eps_;
        }
        break;
    }
    case EpsilonDecayOptionType::EXPONENTIAL:
    {
        eps_ = min_eps_ + (max_eps_ - min_eps_)*std::exp(-epsilon_decay_ * episode);
        break;
    }
    case EpsilonDecayOptionType::CONSTANT_RATE:
    {
        eps_ -= epsilon_decay_;

        if(eps_ < min_eps_){
            eps_ = min_eps_;
        }
        break;
    }
    }
}

uint_t
EpsilonGreedyPolicy::operator()(const table_type& q_map, const state_type& state)const{

    const auto& actions = q_map.find(state)->second;

    std::mt19937 gen(seed_);

    // generate a number in [0, 1]
    std::uniform_real_distribution<> real_dist_(0.0, 1.0);

    if(real_dist_(gen) > eps_){
        // select greedy action with probability 1 - epsilon
        return blaze::argmax(actions);
    }

    //std::mt19937 another_gen(seed_);
    std::uniform_int_distribution<> distrib_(0,  n_actions_ - 1);
    return distrib_(gen);
}

class QLearning
{

public:


    typedef StateAggregationCartPole env_type;
    typedef StateAggregationCartPole::action_type action_type;
    typedef StateAggregationCartPole::state_type state_type;
    typedef EpsilonGreedyPolicy action_selector_type;


    QLearning(const QLearningConfig config, const action_selector_type& selector);


    void actions_before_training_begins(env_type&);
    void actions_after_training_ends(env_type&){}
    void actions_before_episode_begins(env_type&, uint_t /*episode_idx*/){}
    void actions_after_episode_ends(env_type&, uint_t episode_idx){action_selector_.adjust_on_episode(episode_idx);}

    EpisodeInfo on_training_episode(env_type&, uint_t episode_idx);
    void save(std::string /*filename*/)const{}

private:

    ///
    /// \brief config_
    ///
    QLearningConfig config_;

    ///
    /// \brief action_selector_
    ///
    action_selector_type action_selector_;

    ///
    /// \brief q_table_. The tabilar representation of the Q-function
    ///
    table_type q_table_;

    ///
    /// \brief update_q_table_
    /// \param action
    ///
    void update_q_table_(const action_type& action, const state_type& cstate,
                         const state_type& next_state, const  action_type& next_action, real_t reward);

};


QLearning::QLearning(const QLearningConfig config, const action_selector_type& selector)
    :
      config_(config),
      action_selector_(selector),
      q_table_()
{}


void
QLearning::actions_before_training_begins(env_type& env){

    auto states = env.get_states();

    for(auto& state: states){
        q_table_.insert({state, DynVec<real_t>(env.n_actions(), 0.0)});
    }

    //q_table_ = DynMat<real_t>(env.n_states(), env.n_actions(), 0.0);
}

EpisodeInfo
QLearning::on_training_episode(env_type& env, uint_t episode_idx){

    auto start = std::chrono::steady_clock::now();
    EpisodeInfo info;

    // total score for the episode
    auto episode_score = 0.0;
    auto state = env.reset().observation();

    // select an action
    auto action = action_selector_(q_table_, state);

    uint_t itr=0;
    for(;  itr < config_.max_num_iterations_per_episode; ++itr){

        // Take a on_episode
        auto step_type_result = env.step(action);

        auto next_state = step_type_result.observation();
        auto reward = step_type_result.reward();
        auto done = step_type_result.done();

        // accumulate score
        episode_score += reward;

        auto next_action = action_selector_(q_table_, state);
        update_q_table_(action, state, next_state, next_action, reward);
        state = next_state;
        action = next_action;

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

void
QLearning::update_q_table_(const action_type& action, const state_type& cstate,
                                                       const state_type& next_state, const  action_type& /*next_action*/, real_t reward){

    auto current_state_itr = q_table_.find(cstate);

    auto q_current = current_state_itr->second[ action];

    auto next_state_itr = q_table_.find(next_state);
    auto q_next = 0.0;

    if(next_state_itr != q_table_.end()){
        q_next = blaze::max(next_state_itr->second);
    }

    auto td_target = reward + config_.gamma * q_next;
    current_state_itr->second[ action] = q_current + (config_.eta * (td_target - q_current));

}


}


int main(){

    using namespace rl_example_17;

    try{

        Py_Initialize();
        auto main_module = boost::python::import("__main__");
        auto main_namespace = main_module.attr("__dict__");

        // create the environment
        StateAggregationCartPole env("v0", main_namespace, 10);

        // the policy to use
        EpsilonGreedyPolicy policy(EPS, env.n_actions(), EpsilonDecayOptionType::NONE);

        // configuration for the algorithm
        QLearningConfig config;
        config.eta = ALPHA;
        config.gamma = GAMMA;
        config.n_episodes = 50000;
        config.max_num_iterations_per_episode = 10000;

        // the agent to traain
        QLearning algorithm(config, policy);

        RLSerialTrainerConfig trainer_config = {100, 50000, 1.0e-8};

        RLSerialAgentTrainer<StateAggregationCartPole, QLearning> trainer(trainer_config, algorithm);

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
