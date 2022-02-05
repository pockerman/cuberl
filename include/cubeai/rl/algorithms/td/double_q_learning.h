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
class DoubleQLearning final: public TDAlgoBase<EnvTp>, protected with_double_q_table_mixin<TableTp>
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
    /// \brief action_selector_
    ///
    action_selector_type action_selector_;

    ///
    /// \brief update_q_table_
    /// \param action
    ///
    void update_q_table_(const action_type& action, const state_type& cstate,
                         const state_type& next_state, const  action_type& next_action, real_t reward);
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

    // total score for the episode
    auto total_episode_score = 0.0;
    auto state = this->env_ref_().reset().observation();

    uint_t itr=0;
    for(;  itr < this->n_iterations_per_episode(); ++itr){

        // select an action
        auto action = action_selector_(this->q_table(), state);

        if(this->is_verbose()){
            std::cout<<"Episode iteration="<<itr<<" of="<<this->n_iterations_per_episode()<<std::endl;
            std::cout<<"State="<<state<<std::endl;
            std::cout<<"Action="<<action<<std::endl;
        }

        // Take a on_episode
        auto step_type_result = this->env_ref_().step(action);

        auto next_state = step_type_result.observation();
        auto reward = step_type_result.reward();
        auto done = step_type_result.done();

        // accumulate score
        total_episode_score += reward;

        if(!done){
            auto next_action = action_selector_(this->q_table(), state);
            update_q_table_(action, state, next_state, next_action, reward);
            state = next_state;
            action = next_action;
        }
        else{

            update_q_table_(action, state, CubeAIConsts::invalid_size_type(),
                            CubeAIConsts::invalid_size_type(), reward);

            //this->tmp_scores()[current_score_counter_++] = score;

            if(current_score_counter_ >= this->render_env_frequency_){
                current_score_counter_ = 0;
            }

            if(this->is_verbose()){
                std::cout<<"============================================="<<std::endl;
                std::cout<<"Break out from episode="<<this->current_episode_idx()<<std::endl;
                std::cout<<"============================================="<<std::endl;
            }

            break;
        }
    }

    // make any adjustments to the way
    // actions are selected given the experience collected
    // in the episode
    action_selector_.adjust_on_episode(this->current_episode_idx());
    if(current_score_counter_ >= this->render_env_frequency_){
        current_score_counter_ = 0;
    }

    std::cout<<"Finished on_episode="<<this->current_episode_idx()<<std::endl;
}

template <envs::discrete_world_concept EnvTp, typename ActionSelector, typename TableTp>
void
DoubleQLearning<EnvTp, ActionSelector, TableTp>::update_q_table_(const action_type& action, const state_type& cstate,
                                                       const state_type& next_state, const  action_type& next_action, real_t reward){
#ifdef CUBEAI_DEBUG
    assert(action < this->env_ref_().n_actions() && "Inavlid action idx");
    assert(cstate < this->env_ref_().n_states() && "Inavlid state idx");

    if(next_state != CubeAIConsts::invalid_size_type())
        assert(next_state < this->env_ref_().n_states() && "Inavlid next_state idx");

    if(next_action != CubeAIConsts::invalid_size_type())
        assert(next_action < this->env_ref_().n_actions() && "Inavlid next_action idx");
#endif

    // flip a coin 50% of the time we update Q1
    // whilst 50% of the time Q2

    //std::random_device rd;
    std::mt19937 gen(this->seed()); //rd());

    // generate a number in [0, 1]
    std::uniform_real_distribution<> real_dist_(0.0, 1.0);

    // update Q1
    if(real_dist_(gen) <= 0.5){

        // the current qvalue
        auto q_current = with_double_q_table_mixin::q_table_1(cstate, action);
        auto Qsa_next = 0.0;

        if(next_state !=  CubeAIConsts::invalid_size_type() ){
            auto max_act = max_action(with_double_q_table_mixin::q_table_1, next_state, this->env_ref_().n_actions());

            // value of next state
            Qsa_next = with_double_q_table_mixin::q_table_2(next_state, max_act);
         }

         // construct TD target
         auto target = reward + (this->gamma() * Qsa_next);

         // get updated value
         auto new_value = q_current + (this->eta() * (target - q_current));
         with_double_q_table_mixin::q_table_1(cstate, action) = new_value;
    }
    else{

        // the current qvalue
        auto q_current = with_double_q_table_mixin::q_table_2(cstate, action);
        auto Qsa_next = 0.0;

        if(next_state !=  CubeAIConsts::invalid_size_type() ){
            auto max_act = max_action(with_double_q_table_mixin::q_table_2, next_state, this->env_ref_().n_actions());

            // value of next state
            Qsa_next = with_double_q_table_mixin::q_table_1(next_state, max_act);
         }

         // construct TD target
         auto target = reward + (this->gamma() * Qsa_next);

         // get updated value
         auto new_value = q_current + (this->eta() * (target - q_current));
         with_double_q_table_mixin::q_table_2(cstate, action) = new_value;
    }
}

}
}
}
}

#endif // DOUBLE_Q_LEARNING_H
