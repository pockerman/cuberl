#ifndef DUMMY_ALGORITHM_H
#define DUMMY_ALGORITHM_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/rl_algorithm_base.h"
#include "cubeai/rl/algorithms/rl_algo_config.h"
#include "cubeai/utils/iteration_counter.h"

#include "cubeai/base/cubeai_config.h"

#ifdef USE_GYMFCPP
#include "gymfcpp/render_mode_enum.h"
#endif

#include <chrono>
#include <iostream>

namespace cubeai{
namespace rl{
namespace algos{

struct DummyAlgorithmConfig
{
    uint_t n_itrs_per_episode;
};

///
///
///
template<typename EnvType>
class DummyAlgorithm: public RLSolverBase<EnvType>
{
public:

    ///
    /// \brief env_type
    ///
    typedef typename RLSolverBase<EnvType>::env_type env_type;
    typedef std::vector<typename EnvType::action_type> policy_type;

    ///
    /// \brief DummyAgent
    /// \param env
    /// \param config
    ///
    explicit DummyAlgorithm(DummyAlgorithmConfig config);

    ///
    /// \brief actions_before_training_begins. Execute any actions the
    /// algorithm needs before starting the iterations.
    ///
    virtual void actions_before_training_begins(env_type&) override;

    ///
    /// \brief actions_after_training_ends. Actions to execute after
    /// the training iterations have finisehd.
    ///
    virtual void actions_after_training_ends(env_type&) override final {};

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual EpisodeInfo on_training_episode(env_type&, uint_t /*episode_idx*/) override final;

    ///
    /// \brief get_policy
    /// \return
    ///
    policy_type& get_policy(){return actions_;}

    ///
    /// \brief get_policy
    /// \return
    ///
    const policy_type& get_policy()const{return actions_;}

protected:

    ///
    /// \brief itr_counter_
    ///
    cubeai::utils::IterationCounter itr_counter_;

    ///
    /// \brief actions_
    ///
    std::vector<typename EnvType::action_type> actions_;
};

template<typename EnvType>
DummyAlgorithm<EnvType>::DummyAlgorithm(DummyAlgorithmConfig config)
    :
    RLSolverBase<EnvType>(),
    itr_counter_(config.n_itrs_per_episode)
{}

template<typename EnvType>
void
DummyAlgorithm<EnvType>::actions_before_training_begins(env_type& env){
    env.reset();
    actions_.reserve(itr_counter_.max_iterations());
}

template<typename EnvType>
EpisodeInfo
DummyAlgorithm<EnvType>::on_training_episode(env_type& env, uint_t episode_idx){

    auto start = std::chrono::steady_clock::now();
    real_t episode_rewards = 0.0;

    while(itr_counter_.continue_iterations()){

        // sample an action
        auto action = env.sample_action();

        // step the environment
        auto time_step = env.step(action);
        episode_rewards += time_step.reward();
        actions_.push_back(action);

        if(time_step.done()){
           break;
        }    
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<real_t> elapsed_seconds = end-start;

    EpisodeInfo info;
    info.episode_index = episode_idx;
    info.episode_reward = episode_rewards;
    info.episode_iterations = itr_counter_.current_iteration_index();
    info.total_time = elapsed_seconds;
    return info;

}

}
}
}

#endif // DUMMY_ALGORITHM_H
