#ifndef RL_SERIAL_AGENT_TRAINER_H
#define RL_SERIAL_AGENT_TRAINER_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/iterative_algorithm_result.h"
#include "cubeai/base/iterative_algorithm_controller.h"

#include <boost/noncopyable.hpp>
#include <vector>

namespace cubeai {
namespace rl {


///
/// \brief The RLSerialTrainerConfig struct. Configuration
/// struct for the serial RL agent trainer
///
struct RLSerialTrainerConfig
{
    uint_t output_msg_frequency;
    uint_t n_episodes;
    real_t tolerance;
};


///
/// \detailed The RLSerialAgentTrainer class handles the training
/// for serial reinforcement learning agents
///
template<typename EnvType, typename AgentType>
class RLSerialAgentTrainer: private boost::noncopyable
{
public:

    typedef EnvType env_type;
    typedef AgentType agent_type;

    ///
    /// \brief RLSerialAgentTrainer
    /// \param config
    /// \param agent
    ///
    RLSerialAgentTrainer(RLSerialTrainerConfig& config, agent_type& agent);

    ///
    /// \brief train Iterate to train the agent on the given
    /// environment
    ///
    virtual IterativeAlgorithmResult train(env_type& env);

    ///
    /// \brief actions_before_training_begins.  Execute any actions
    /// the algorithm needs before starting the episode
    ///
    virtual void actions_before_training_begins(env_type&);

    ///
    /// \brief actions_before_episode_begins. Execute any actions the algorithm needs before
    /// starting the episode
    ///
    virtual void actions_before_episode_begins(env_type&, uint_t);

    ///
    /// \brief  actions_after_episode_ends. Execute any actions the algorithm needs after
    /// ending the episode
    ///
    virtual void actions_after_episode_ends(env_type&, uint_t);

    ///
    /// \brief actions_after_training_ends. Execute any actions the algorithm needs after
    /// the iterations are finished
    ///
    virtual void actions_after_training_ends(env_type&);

protected:

    ///
    /// \brief itr_ctrl_ Handles the iteration over the
    /// episodes
    ///
    IterativeAlgorithmController itr_ctrl_;

    ///
    /// \brief agent_
    ///
    agent_type& agent_;

    ///
    /// \brief total_reward_per_episode_
    ///
    std::vector<real_t> total_reward_per_episode_;

    ///
    /// \brief n_itrs_per_episode_ Holds the number of iterations
    /// performed per training episode
    ///
    std::vector<uint_t> n_itrs_per_episode_;

};

template<typename EnvType, typename AgentType>
void
RLSerialAgentTrainer<EnvType, AgentType>::actions_before_training_begins(env_type& env){

    agent_.actions_before_training_begins(env);
    total_reward_per_episode_.clear();
    n_itrs_per_episode_.clear();

    total_reward_per_episode_.reserve(itr_ctrl_.get_max_iterations());
    n_itrs_per_episode_.reserve(itr_ctrl_.get_max_iterations());
}

template<typename EnvType, typename AgentType>
void
RLSerialAgentTrainer<EnvType, AgentType>::actions_before_episode_begins(env_type& env, uint_t episode_idx){
   agent_.actions_before_episode_begins(env, episode_idx);
}

template<typename EnvType, typename AgentType>
void
RLSerialAgentTrainer<EnvType, AgentType>::actions_after_episode_ends(env_type& env, uint_t episode_idx){
    agent_.actions_after_episode_ends(env, episode_idx);
}

template<typename EnvType, typename AgentType>
void
RLSerialAgentTrainer<EnvType, AgentType>::actions_after_training_ends(env_type& env){
    agent_.actions_after_training_ends(env);
}

template<typename EnvType, typename AgentType>
IterativeAlgorithmResult
RLSerialAgentTrainer<EnvType, AgentType>::train(env_type& env){

    this->actions_before_training_begins(env);

    while(itr_ctrl_.continue_iterations()){

        this->actions_before_episode_begins(env, itr_ctrl_.get_current_iteration());
        auto episode_info = agent_.on_training_episode(env, itr_ctrl_.get_current_iteration());

        total_reward_per_episode_.push_back(episode_info.episode_reward);
        n_itrs_per_episode_.push_back(episode_info.episode_iterations);
        this->actions_after_episode_ends(env, itr_ctrl_.get_current_iteration());
    }

    this->actions_after_training_ends(env);

    return itr_ctrl_.get_state();


    /*        counter = 0
            while self._itr_ctrl.continue_itrs():

                if self.trainer_config.output_msg_frequency != -1:
                    remains = counter % self.trainer_config.output_msg_frequency
                    if remains == 0:
                        print("{0}: Episode {1} of {2}, ({3}% done)".format(INFO, self.current_episode_index,
                                                                            self.itr_control.n_max_itrs,
                                                                            (self.itr_control.current_itr_counter / self.itr_control.n_max_itrs) * 100.0))
                self.actions_before_episode_begins(env, **options)
                episode_info = self.agent.on_training_episode(env, self.current_episode_index, **options)
                self.rewards.append(episode_info.episode_reward)
                self.iterations_per_episode.append(episode_info.episode_iterations)

                if "break_training" in episode_info.info and \
                        episode_info.info["break_training"] is True:
                    self.break_training_flag = True

                self.actions_after_episode_ends(env, **options)
                counter += 1

                # check if the break training flag
                # has been set and break
                if self.break_training_flag:
                    print("{0}: On Episode {1} the break training "
                          "flag was set. Stop training".format(INFO, self.current_episode_index))

                    # if we get here then assume we have converged
                    self._itr_ctrl.residual = self.trainer_config.tolerance * 1.0e-2
                    break

            self.actions_after_training_ends(env, **options)

            # update the control result
            itr_ctrl_rsult.n_itrs = self._itr_ctrl.current_itr_counter
            itr_ctrl_rsult.residual = self._itr_ctrl.residual

            return itr_ctrl_rsult*/

}


}
}

#endif // RL_SERIAL_AGENT_TRAINER_H
