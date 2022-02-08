#ifndef DOUBLE_Q_LEARNING_H
#define DOUBLE_Q_LEARNING_H

///
/// Implementation of tabular double q-learning algorithm
///
///

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_consts.h"
#include "cubeai/rl/algorithms/td/td_algo_base.h"
#include "cubeai/rl/rl_mixins.h"
#include "cubeai/rl/worlds/envs_concepts.h"

#include <random>

namespace cubeai{
namespace rl{
namespace algos{
namespace td{


///
/// \brief The class DoubleQLearning. Simple tabular implemtation
/// of double q-learning algorithm.
///
template<envs::discrete_world_concept EnvTp, typename ActionSelector, typename TableTp>
class DoubleQLearning final: public TDAlgoBase<EnvTp>, protected with_double_q_table_mixin<TableTp>, protected with_double_q_table_max_action_mixin
{
public:


    ///
    /// \brief env_t
    ///
    typedef typename TDAlgoBase<EnvTp>::env_type env_type;

    ///
    /// \brief action_t
    ///
    typedef typename TDAlgoBase<EnvTp>::action_type action_type;

    ///
    /// \brief state_t
    ///
    typedef typename TDAlgoBase<EnvTp>::state_type state_type;

    ///
    /// \brief action_selector_t
    ///
    typedef ActionSelector action_selector_type;

    ///
    /// \brief Constructor
    ///
    DoubleQLearning(TDAlgoConfig config, env_type& env, const ActionSelector& selector);

    ///
    /// \brief on_episode. Performs the iterations for
    /// one training episode
    ///
    virtual void on_episode()override final;

    ///
    ///
    ///
    virtual void actions_before_training_episodes()override final;

private:

    ///
    /// \brief current_score_counter_
    ///
    uint_t current_score_counter_;

    ///
    /// \brief action_selector_. This is typically the policy
    /// we use to select actions. e.g. epsilon-greedy or softmax
    ///
    action_selector_type action_selector_;

    ///
    /// \brief update_q_table_
    /// \param action
    ///
    void update_q_table_(const action_type& action, const state_type& cstate,
                         const state_type& next_state, real_t reward);

};

template <envs::discrete_world_concept EnvTp, typename ActionSelector, typename TableTp>
DoubleQLearning<EnvTp, ActionSelector, TableTp>::DoubleQLearning(TDAlgoConfig config, env_type& env, const ActionSelector& selector)
    :
     TDAlgoBase<EnvTp>(config, env),
     with_double_q_table_mixin<TableTp>(),
     action_selector_(selector)
{}


template<envs::discrete_world_concept EnvTp, typename ActionSelector, typename TableTp>
void
DoubleQLearning<EnvTp, ActionSelector, TableTp>::actions_before_training_episodes(){

    this->TDAlgoBase<EnvTp>::actions_before_training_episodes();

    auto states = this->env_ref_().get_states();
    this->with_double_q_table_mixin<TableTp>::initialize(states, this->env_ref_().n_actions(), 0.0);
}

template<envs::discrete_world_concept EnvTp, typename ActionSelector, typename TableTp>
void
DoubleQLearning<EnvTp, ActionSelector, TableTp>::on_episode(){

     // typedef typename with_double_q_table_mixin<TableTp>::state_type discrete_state_type;

    // total score for the episode
    auto total_episode_reward = 0.0;
    auto state = this->env_ref_().reset().observation();

    uint_t itr=0;
    for(;  itr < this->n_iterations_per_episode(); ++itr){

        // select an action
        auto action = action_selector_(this->with_double_q_table_mixin<TableTp>::q_table_1,
                                       this->with_double_q_table_mixin<TableTp>::q_table_2, state);

        // Take an action on the environment
        auto step_type_result = this->env_ref_().step(action);

        auto next_state = step_type_result.observation();
        auto reward = step_type_result.reward();
        auto done = step_type_result.done();

        // accumulate score
        total_episode_reward += reward;

        // update the table
        update_q_table_(action, state, next_state, reward);
        state = next_state;

        if(done){
            break;
        }
    }

    this->get_iterations()[this->current_episode_idx()] = itr;
    this->get_rewards()[this->current_episode_idx()] = total_episode_reward;

    // make any adjustments to the way
    // actions are selected given the experience collected
    // in the episode
    action_selector_.adjust_on_episode(this->current_episode_idx());
    if(current_score_counter_ >= this->render_env_frequency_){
        current_score_counter_ = 0;
    }
}

template <envs::discrete_world_concept EnvTp, typename ActionSelector, typename TableTp>
void
DoubleQLearning<EnvTp, ActionSelector, TableTp>::update_q_table_(const action_type& action, const state_type& cstate,
                                                                 const state_type& next_state, real_t reward){

//#ifdef CUBEAI_DEBUG
//    assert(action < this->env_ref_().n_actions() && "Inavlid action idx");
//    assert(cstate < this->env_ref_().n_states() && "Inavlid state");
//
//    if(next_state != CubeAIConsts::invalid_size_type())
//        assert(next_state < this->env_ref_().n_states() && "Inavlid next_state idx");
//#endif


    // flip a coin 50% of the time we update Q1
    // whilst 50% of the time Q2
    std::mt19937 gen(this->seed()); //rd());

    // generate a number in [0, 1]
    std::uniform_real_distribution<> real_dist_(0.0, 1.0);

    // update Q1
    if(real_dist_(gen) <= 0.5){

        // the current qvalue
        auto q_current = this->with_double_q_table_mixin<TableTp>::template get<1>(cstate, action);
        auto Qsa_next = 0.0;

        //if(this->env_ref_().is_valid_state(next_state)){
            auto max_act = this->with_double_q_table_max_action_mixin::max_action(this->with_double_q_table_mixin<TableTp>::q_table_1,
                                                                                  next_state, this->env_ref_().n_actions());

            // value of next state
            Qsa_next = this->with_double_q_table_mixin<TableTp>::template get<2>(next_state, max_act);
         //}

         // construct TD target
         auto target = reward + (this->gamma() * Qsa_next);

         // get updated value
         auto new_value = q_current + (this->eta() * (target - q_current));
         this->with_double_q_table_mixin<TableTp>::template set<1>(cstate, action, new_value);
    }
    else{

        // the current qvalue
        auto q_current = this->with_double_q_table_mixin<TableTp>::template get<2>(cstate, action);
        auto Qsa_next = 0.0;

        //if(this->env_ref_().is_valid_state(next_state)){
            auto max_act = this->with_double_q_table_max_action_mixin::max_action(this->with_double_q_table_mixin<TableTp>::q_table_2,
                                                                                  next_state, this->env_ref_().n_actions());

            // value of next state
            Qsa_next = this->with_double_q_table_mixin<TableTp>::template get<1>(next_state, max_act);
         //}

         // construct TD target
         auto target = reward + (this->gamma() * Qsa_next);

         // get updated value
         auto new_value = q_current + (this->eta() * (target - q_current));
         this->with_double_q_table_mixin<TableTp>::template set<2>(cstate, action, new_value);
    }
}

}
}
}
}

#endif // DOUBLE_Q_LEARNING_H
