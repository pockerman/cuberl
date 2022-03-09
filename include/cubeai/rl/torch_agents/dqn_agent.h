#ifndef DQN_AGENT_H
#define DQN_AGENT_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/rl/torch_agents/torch_agent_base.h"
#include "cubeai/rl/episode_result_info.h"

namespace cubeai{
namespace rl{
namespace pytorch{

template<typename EnvType, typename ActionSelector>
class DQNAgent: public TorchAgentBase<EnvType>
{
public:

    typedef typename TorchAgentBase<EnvType>::env_type env_type;
    typedef typename TorchAgentBase<EnvType>::time_step_type time_step_type;
    typedef ActionSelector action_selector_type;

    ///
    /// \brief DQNAgent
    /// \param config
    ///
    DQNAgent(const TorchAgentConfig& config);

    ///
    /// \brief on_training_episode
    /// \param env
    /// \param episode_idx
    /// \return
    ///
    virtual EpisodeResultInfo on_training_episode(env_type& env, uint_t episode_idx);

protected:

    ///
    /// \brief config_
    ///
    TorchAgentConfig config_;

    ///
    /// \brief current_time_step_
    ///
    time_step_type current_time_step_;

    ///
    /// \brief action_selector_
    ///
    action_selector_type action_selector_;

};

template<typename EnvType, typename ActionSelector>
EpisodeResultInfo
DQNAgent<EnvType, ActionSelector>::on_training_episode(env_type& env, uint_t episode_idx){

    for(uint_t itr=0; itr < config_.n_itrs_per_episode; ++itr){

        // get the model predictions
        auto model_vals = this->model_.forward(current_time_step_.observation());

        // choose an action. The action selector
        // is able to resolve the action type
        auto action = action_selector_.choose_action(model_vals);


    }


}

}
}
}

#endif
#endif // DQN_AGENT_H
