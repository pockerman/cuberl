#ifndef Q_LEARNING_H
#define Q_LEARNING_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/td/td_algo_base.h"
#include "cubeai/rl/worlds/envs_concepts.h"
#include "cubeai/rl/episode_info.h"
#include "cubeai/maths/matrix_utilities.h"
#include "cubeai/base/cubeai_config.h"

#include "rlenvs/utils/io/csv_file_writer.h"
#include "rlenvs/rlenvs_consts.h"

#ifdef CUBERL_DEBUG
#include <cassert>
#endif

#include <chrono>

namespace cuberl {
namespace rl{
namespace algos {
namespace td {

///
/// \brief The QLearningConfig struct
///
struct QLearningConfig
{
    uint_t n_episodes;
    real_t tolerance;
    real_t gamma;
    real_t eta;
    uint_t max_num_iterations_per_episode;
    std::string path{rlenvscpp::consts::INVALID_STR};
};


///
/// \brief The QLearning class. Table based implementation
/// of the Q-learning algorithm using epsilon-greedy policy.
/// The implementation also allows for exponential decay
/// of the used epsilon
///
template<envs::discrete_world_concept EnvTp, typename ActionSelector>
class QLearning final: public TDAlgoBase<EnvTp>
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
    QLearning(const QLearningConfig config, const ActionSelector& selector);

    ///
    /// \brief actions_before_training_begins. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_begins(env_type&);

    ///
    /// \brief actions_after_training_ends. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_ends(env_type&);

    ///
    /// \brief actions_before_training_episode
    ///
    virtual void actions_before_episode_begins(env_type&, uint_t /*episode_idx*/){}

    ///
    /// \brief actions_after_training_episode
    ///
    virtual void actions_after_episode_ends(env_type&, uint_t episode_idx,
                                            const EpisodeInfo& /*einfo*/);

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual EpisodeInfo on_training_episode(env_type&, uint_t episode_idx);

    ///
    ///
    ///
    void save(std::string filename)const;

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
    DynMat<real_t> q_table_;

    ///
    /// \brief update_q_table_
    /// \param action
    ///
    void update_q_table_(const action_type& action, const state_type& cstate,
                         const state_type& next_state, const  action_type& next_action, real_t reward);

};

template <envs::discrete_world_concept EnvTp, typename ActionSelector>
QLearning<EnvTp, ActionSelector>::QLearning(const QLearningConfig config, 
											const ActionSelector& selector)
    :
      TDAlgoBase<EnvTp>(),
      config_(config),
      action_selector_(selector),
      q_table_()
{}

template<envs::discrete_world_concept EnvTp, typename ActionSelector>
void
QLearning<EnvTp, ActionSelector>::actions_before_training_begins(env_type& env){
    q_table_ = DynMat<real_t>(env.n_states(), env.n_actions());

    for(uint_t i=0; i < env.n_states(); ++i)
        for(uint_t j=0; j < env.n_actions(); ++j)
            q_table_(i, j) = 0.0;

}

template<envs::discrete_world_concept EnvTp, typename ActionSelector>
void
QLearning<EnvTp, ActionSelector>::actions_after_training_ends(env_type&){

    if(config_.path != rlenvscpp::consts::INVALID_STR){
        save(config_.path);
    }
}


template<envs::discrete_world_concept EnvTp, typename ActionSelector>
EpisodeInfo
QLearning<EnvTp, ActionSelector>::on_training_episode(env_type& env, uint_t episode_idx){

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

        if(!done){
            auto next_action = action_selector_(q_table_, state);
            update_q_table_(action, state, next_state, next_action, reward);
            state = next_state;
            action = next_action;
        }
        else{

            update_q_table_(action, state, 
			                rlenvscpp::consts::INVALID_ID,
                            rlenvscpp::consts::INVALID_ID, 
							reward);


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

template<envs::discrete_world_concept EnvTp, typename ActionSelector>
void
QLearning<EnvTp, ActionSelector>::actions_after_episode_ends(env_type&, uint_t episode_idx,
                                                            const EpisodeInfo& /*einfo*/){
    action_selector_.on_episode(episode_idx);
}

template<envs::discrete_world_concept EnvTp, typename ActionSelector>
void
QLearning<EnvTp, ActionSelector>::save(std::string /*filename*/)const{

    /*CSVWriter file_writer(filename, ',', true);
    std::vector<std::string> col_names(1 + q_table_.columns());
    col_names[0] = "state_index";

    for(uint_t i = 0; i< q_table_.columns(); ++i){
        col_names[i + 1] = "action_" + std::to_string(i);
    }

    file_writer.write_column_names(col_names);

    for(uint_t s=0; s < q_table_.rows(); ++s){
        auto actions = maths::get_row(q_table_, s);
        auto row = std::make_tuple(s, actions);
        file_writer.write_row(row);
    }*/

}

template <envs::discrete_world_concept EnvTp, typename ActionSelector>
void
QLearning<EnvTp, ActionSelector>::update_q_table_(const action_type& action, const state_type& cstate,
												  const state_type& next_state, 
												  const  action_type& /*next_action*/, real_t reward){

    auto q_current = q_table_(cstate, action);
    auto q_next = next_state != rlenvscpp::consts::INVALID_ID ? cuberl::maths::get_row_max(q_table_, next_state) : 0.0;


    auto td_target = reward + config_.gamma * q_next;
    q_table_(cstate, action) = q_current + (config_.eta * (td_target - q_current));

}



}
}
}
}

#endif // Q_LEARNING_H
