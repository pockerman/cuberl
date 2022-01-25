#ifndef DOUBLE_Q_LEARNING_H
#define DOUBLE_Q_LEARNING_H

/**
  * Implementation of tabular double q-learning algorithm
  * */

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/td/td_algo_base.h"
#include "cubeai/rl/rl_mixins.h"


#if __cplusplus >= 202002L
#include "cubeai/rl/worlds/envs_concepts.h"
#endif

namespace cubeai{
namespace rl{
namespace algos{
namespace td{


/**
 * \brief The class DoubleQLearning. Simple tabular implemtation
 * of double q-learning algorithm.
 */
template<envs::discrete_world_concept EnvTp, typename ActionSelector>
class DoubleQLearning final: public TDAlgoBase<EnvTp>, protected with_double_q_table_mixin
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

template <envs::discrete_world_concept EnvTp, typename ActionSelector>
DoubleQLearning<EnvTp, ActionSelector>::DoubleQLearning(TDAlgoConfig config, env_type& env, const ActionSelector& selector)
    :
     TDAlgoBase<EnvTp>(config, env),
     with_double_q_table_mixin(),
     action_selector_(selector)
{}


template<envs::discrete_world_concept EnvTp, typename ActionSelector>
void
DoubleQLearning<EnvTp, ActionSelector>::on_episode(){

    // total score for the episode
    auto total_episode_score = 0.0;
    auto state = this->env_ref_().reset().observation();

    // select an action
    auto action = action_selector_(this->q_table(), state);

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
        score += reward;

        if(!done){
            auto next_action = action_selector_(this->q_table(), state);
            update_q_table_(action, state, next_state, next_action, reward);
            state = next_state;
            action = next_action;
        }
        else{

            update_q_table_(action, state, CubeAIConsts::invalid_size_type(),
                            CubeAIConsts::invalid_size_type(), reward);

            this->tmp_scores()[current_score_counter_++] = score;

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

template <envs::discrete_world_concept EnvTp, typename ActionSelector>
void
DoubleQLearning<EnvTp, ActionSelector>::update_q_table_(const action_type& action, const state_type& cstate,
                                                       const state_type& next_state, const  action_type& next_action, real_t reward){
#ifdef CUBEAI_DEBUG
    assert(action < this->env_ref_().n_actions() && "Inavlid action idx");
    assert(cstate < this->env_ref_().n_states() && "Inavlid state idx");

    if(next_state != CubeAIConsts::invalid_size_type())
        assert(next_state < this->env_ref_().n_states() && "Inavlid next_state idx");

    if(next_action != CubeAIConsts::invalid_size_type())
        assert(next_action < this->env_ref_().n_actions() && "Inavlid next_action idx");
#endif

    auto q_current = this->q_table()[cstate][action];
    auto q_next = next_state != CubeAIConsts::invalid_size_type() ? blaze::max(this->q_table()[next_state]) : 0.0;
    auto td_target = reward + this->gamma() * q_next;
    this->q_table()[cstate][action] = q_current + (this->eta() * (td_target - q_current));

}

}
}
}
}

#endif // DOUBLE_Q_LEARNING_H
